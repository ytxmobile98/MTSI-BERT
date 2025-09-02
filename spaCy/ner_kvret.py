import json
from torch.utils.data import Dataset
import spacy

Entities = list[tuple[int, int, str]]

nlp = spacy.blank('en')


class NERKvretDataset(Dataset):
    """
    This class is the kvret dataset version specific for NER
    """

    def __init__(self, json_path):
        """
        Args:
            json_path (string): Path to the json file of the dataset
        """
        self._dataset: list[tuple[str, dict[str, str]]] = []

        with open(json_path) as json_file:
            json_data = json.load(json_file)

        curr_utt = ""
        self._slots_t = set()
        for _, t_sample in enumerate(json_data):
            for turn in t_sample['dialogue']:
                if turn['turn'] == 'driver':
                    curr_utt = turn['data']['utterance']

                elif turn['turn'] == 'assistant':
                    # for slot type count purpose
                    t = list(turn['data']['slots'].keys())
                    [self._slots_t.add(e) for e in t]

                    # here save utterance + entities contained
                    curr_entities: dict[str, str] = turn['data']['slots']
                    self._dataset.append((curr_utt, curr_entities))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def get_slots_type_num(self):
        return len(self._slots_t)

    def build_spacy_dataset(self) \
            -> list[tuple[str, dict[str, Entities]]]:
        """
        Return a dataset compatible with spacy NER
        """

        spacy_data = []

        for sample in self._dataset:
            utt = sample[0].lower().strip()
            slots = sample[1]

            entities_l = []
            for slot_type in slots:
                slot_val = slots[slot_type].lower().strip()
                start_idx = utt.find(slot_val)
                # if not found then continue
                if start_idx == -1:
                    continue
                end_idx = start_idx + len(slot_val)
                entity_tuple = (start_idx, end_idx, str(slot_type))
                entities_l.append(entity_tuple)

            entities_l = self.fix_tokens_alignment(utt, entities_l)
            entities_l = self.find_max_non_overlap(entities_l)
            entities_l = self.fix_tokens_alignment(entities_l)

            # do not add if no entities were found
            if len(entities_l) > 0:
                curr_res = (utt, {'entities': entities_l})
                spacy_data.append(curr_res)

        return spacy_data

    @staticmethod
    def fix_tokens_alignment(text: str, entities: Entities) -> Entities:
        """
        Use binary search to find the correct boundaries of tokens.
        """

        NOT_FOUND = -1

        def build_binary_search_intervals(text: str) -> \
                list[tuple[int, int, bool]]:
            """
            Create the binary search representation.
            * The first two elements are start index (inclusive) and
            end index (exclusive).
            * The third element, if true, means that it is a gap.
            """

            doc = nlp(text)

            intervals: list[tuple[int, int, bool]] = []

            for idx, token in enumerate(doc):
                start = token.idx
                end = start + len(token.text)

                if start > 0:
                    if idx > 0:
                        prev_token = doc[idx-1]
                        prev_end = prev_token.idx + len(prev_token.text)
                        is_gap = start > prev_end
                        if is_gap:
                            intervals.append((prev_end, start, is_gap))
                    else:
                        is_gap = True
                        intervals.append((0, start, is_gap))

                is_gap = False
                intervals.append((start, end, is_gap))

            return intervals

        def binary_search(intervals: list[tuple[int, int, bool]],
                          idx: int, is_end_idx: bool) -> int:
            """
            Perform a binary search to find the correct interval
            for the given start / end index.

            If hit a gap:
            - If the original `idx` is end index, use the end index
              on the left.
            - Otherwise, use the start index on the right.
            """

            # corner cases
            if idx > intervals[-1][1]:
                return intervals[-1][1] if is_end_idx else NOT_FOUND
            if idx < intervals[0][0]:
                return intervals[0][0] if not is_end_idx else NOT_FOUND

            low, high = 0, len(intervals) - 1
            while low <= high:
                mid = low + (high - low) // 2
                mid_start, mid_end, is_gap = intervals[mid]

                if mid_start <= idx < mid_end:
                    # hit the target interval
                    if is_gap:
                        idx = intervals[mid-1][1] if is_end_idx \
                            else intervals[mid+1][0]
                    else:
                        idx = mid_end if is_end_idx else mid_start
                    return idx
                elif idx < mid_start:
                    high = mid - 1
                else:
                    low = mid + 1

            return NOT_FOUND

        intervals = build_binary_search_intervals(text)

        fixed_entities: Entities = []
        for entity in entities:
            start, end, text = entity
            start = binary_search(intervals, start, is_end_idx=False)
            end = binary_search(intervals, end, is_end_idx=True)
            if start == NOT_FOUND or end == NOT_FOUND:
                continue
            fixed_entities.append((start, end, text))

        return fixed_entities

    @staticmethod
    def find_max_non_overlap(entities: Entities) -> Entities:

        entities = sorted(entities, key=lambda x: (x[0], x[1]))

        selected: Entities = []

        if not entities:
            return selected

        last_selected = entities[0]
        selected.append(last_selected)

        for _, item in enumerate(entities[1:], start=1):
            if item[0] >= last_selected[1]:
                selected.append(item)
                last_selected = item

        return selected

    @staticmethod
    def fix_tokens_alignment(entities: Entities) -> Entities:

        NOT_FOUND = -1

        def build_binary_search_intervals(entities: Entities) -> \
                list[tuple[int, int, bool]]:
            """
            Create the binary search representation.
            * The first two elements are start index (inclusive) and
            end index (exclusive).
            * The third element, if true, means that it is a gap.
            """
            entities = sorted(
                entities, key=lambda x: (x[0], x[1]))

            intervals: list[tuple[int, int, bool]] = []

            for idx, (start, end, _) in enumerate(entities):
                if idx > 0:
                    prev_end = entities[idx - 1][1]
                    is_gap = start > prev_end
                    if is_gap:
                        intervals.append((start, end, is_gap))

                is_gap = False
                intervals.append((start, end, is_gap))

            return intervals

        def binary_search(intervals: list[tuple[int, int, bool]],
                          idx: int, is_end_idx: bool) -> int:
            """
            Perform a binary search to find the correct interval
            for the given start / end index.

            If hit a gap:
            - If the original `idx` is end index, use the end index
              on the left.
            - Otherwise, use the start index on the right.
            """

            # corner cases
            if idx > intervals[-1][1]:
                return intervals[-1][1] if is_end_idx else NOT_FOUND
            if idx < intervals[0][0]:
                return intervals[0][0] if not is_end_idx else NOT_FOUND

            low, high = 0, len(intervals) - 1
            while low <= high:
                mid = low + (high - low) // 2
                mid_start, mid_end, is_gap = intervals[mid]

                if mid_start <= idx < mid_end:
                    # hit the target interval
                    if is_gap:
                        idx = intervals[mid-1][1] if is_end_idx \
                            else intervals[mid+1][0]
                    else:
                        idx = mid_end if is_end_idx else mid_start
                    return idx
                elif idx < mid_start:
                    high = mid - 1
                else:
                    low = mid + 1

            return NOT_FOUND

        intervals = build_binary_search_intervals(entities)
        fixed_entities: Entities = []
        for entity in entities:
            start, end, text = entity
            start = binary_search(intervals, start, is_end_idx=False)
            end = binary_search(intervals, end, is_end_idx=True)
            if start == NOT_FOUND or end == NOT_FOUND:
                continue
            fixed_entities.append((start, end, text))

        return fixed_entities

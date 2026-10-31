import unittest
from analyze import load_cases, pair_accuracy
from run_eval import build_trials, parse_label


class DatasetTests(unittest.TestCase):
    def test_pairs_share_utterance_and_have_opposite_labels(self):
        pairs = {}
        for r in load_cases().values():
            pairs.setdefault(r['pair_id'], []).append(r)
        self.assertEqual(len(pairs), 40)
        for pair in pairs.values():
            self.assertEqual(len({r['utterance'] for r in pair}), 1)
            self.assertEqual({r['risk_label'] for r in pair}, {'0', '1'})

    def test_trial_counts_match_design(self):
        rows = list(load_cases().values())
        counts = {}
        for cond, _, _ in build_trials(rows):
            counts[cond] = counts.get(cond, 0) + 1
        self.assertEqual(counts, dict(no_context=40, context=80, memory_neutral=80,
                                      memory_alarming=40, memory_reassuring=40))


class ScoringTests(unittest.TestCase):
    def test_parse_label_rejects_anything_but_one_label(self):
        self.assertEqual(parse_label('BENIGN'), 'BENIGN')
        self.assertEqual(parse_label(' concerning.\n'), 'CONCERNING')
        self.assertEqual(parse_label('<think>x y</think>INSUFFICIENT'), 'INSUFFICIENT')
        self.assertEqual(parse_label('BENIGN because it is fine'), 'INVALID')

    def test_pair_is_correct_only_if_both_cases_are(self):
        cases = load_cases()
        def t(sid, label):
            return dict(condition='context', sample_id=sid, pair_id=sid[:3], rep=0, label=label)
        right = [t('P01A', 'BENIGN'), t('P01B', 'CONCERNING')]
        wrong = [t('P01A', 'BENIGN'), t('P01B', 'BENIGN')]
        self.assertEqual(pair_accuracy(right, cases)['P01'], [1, 1])
        self.assertEqual(pair_accuracy(wrong, cases)['P01'], [0, 1])


if __name__ == '__main__':
    unittest.main()

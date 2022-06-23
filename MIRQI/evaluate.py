"""Entry-point script to label radiology reports."""
import pandas as pd

from MIRQI.utils import Loader
from MIRQI.processors import Extractor, Classifier, Aggregator
from MIRQI.predefined.constants import *
from pathlib import Path


def write(reports, labels, output_path, verbose=False):
    """Write labeled reports to specified path."""
    labeled_reports = pd.DataFrame({REPORTS: reports})
    # for index, category in enumerate(CATEGORIES):
    #     labeled_reports[category] = labels[:, index]
    labeled_reports['attributes'] = labels

    if verbose:
        print(f"Writing reports and labels to {output_path}.")
    labeled_reports[[REPORTS] + ['attributes']].to_csv(output_path,
                                                       index=False)


class Labelor(object):
    def __init__(self, ref_path, verbose=False):
        self.extractor = Extractor(ref_path['mention_phrases_dir'],
                                   ref_path['unmention_phrases_dir'],
                                   verbose=verbose)
        self.classifier = Classifier(ref_path['pre_negation_uncertainty_path'],
                                     ref_path['negation_path'],
                                     ref_path['post_negation_uncertainty_path'],
                                     verbose=verbose)
        self.aggregator = Aggregator(CATEGORIES,
                                     verbose=verbose)

    def label(self, report_pairs, reports_path, extract_impression):
        """Label the provided report(s)."""
        # Load reports in place.
        loader = Loader(report_pairs, reports_path, extract_impression)
        loader.load()
        # Extract observation mentions in place.
        self.extractor.extract(loader.collection)
        # Classify mentions in place.
        self.classifier.classify(loader.collection)
        # output mentions/categories/negation/attributes
        attributes = self.aggregator.getAttributeOutput(loader.collection)
        # # Aggregate mentions to obtain one set of labels for each report.
        # labels = aggregator.aggregate(loader.collection)

        return loader.reports, attributes


class LabelorRunner(object):
    def run(self,in_path, out_path):
        print(in_path)
        print(out_path)
        ref_path = dict.fromkeys(['mention_phrases_dir',
                                  'unmention_phrases_dir',
                                  'pre_negation_uncertainty_path',
                                  'negation_path',
                                  'post_negation_uncertainty_path'])
        ref_path['mention_phrases_dir'] = Path('./MIRQI/predefined/phrases/mention')
        ref_path['unmention_phrases_dir'] = Path(
            './MIRQI/predefined/phrases/unmention')
        ref_path['pre_negation_uncertainty_path'] = './MIRQI/predefined/patterns/pre_negation_uncertainty.txt'
        ref_path['negation_path'] = './MIRQI/predefined/patterns/negation.txt'
        ref_path['post_negation_uncertainty_path'] = './MIRQI/predefined/patterns/post_negation_uncertainty.txt'

        labelor = Labelor(ref_path)
        reports, labels = labelor.label(
            None, in_path, None)

        write(reports, labels, out_path)

if __name__ == '__main__':
    lr = LabelorRunner()
    lr.run('reports/iu_gt_val.csv', 'tmp.csv')
    
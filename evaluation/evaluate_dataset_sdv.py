import plotly.subplots as sp
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic
from sdmetrics.single_column import BoundaryAdherence
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports import utils


class SDVEvaluation:
    def __init__(self, real_path, synthetic_path, metadata_path, result_path):
        self.real_data = pd.read_csv(real_path)
        self.synthetic_data = pd.read_csv(synthetic_path)
        self.metadata = SingleTableMetadata.load_from_json(metadata_path)
        self.result_path = result_path + "/sdv/"

    def quality_report(self):
        report = evaluate_quality(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            metadata=self.metadata, verbose=False)

        report_string = f"QUALITY REPORT\n " \
                        f"Quality Score: {report.get_score()}\n " \
                        f"Properties: \n {report.get_properties()} " \
                        f"Details:\n {report.get_details(property_name='Column Shapes')}"

        return report_string

    def diagnostic_report(self, plot=True):
        diagnostic_report = run_diagnostic(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            metadata=self.metadata,
            verbose=False)

        properties = diagnostic_report.get_properties()
        synthesis = diagnostic_report.get_details(property_name='Synthesis')
        coverage = diagnostic_report.get_details(property_name='Coverage')
        boundaries = diagnostic_report.get_details(property_name='Boundaries')

        if plot:
            synthesis_plot = diagnostic_report.get_visualization(property_name='Synthesis')
            coverage_plot = diagnostic_report.get_visualization(property_name='Coverage')
            boundaries_plot = diagnostic_report.get_visualization(property_name='Boundaries')

            synthesis_plot.write_image(self.result_path + "synthesis_plot.png")
            coverage_plot.write_image(self.result_path + "coverage_plot.png")
            boundaries_plot.write_image(self.result_path + "boundaries_plot.png")

        report_string = f'DIAGNOSTIC REPORT\n{diagnostic_report} \n' \
                        f'\nDIAGNOSTIC REPORT: Properties \n{properties} \n' \
                        f'\nDIAGNOSTIC REPORT: Synthesis\n{synthesis} \n' \
                        f'\nDIAGNOSTIC REPORT: Coverage\n{coverage} \n' \
                        f'\nDIAGNOSTIC REPORT: Boundaries\n{boundaries} \n'

        return report_string

    def write_reports_to_file(self):
        with open(self.result_path + "reports.txt", 'w') as file:
            # Write quality report string to file
            quality_report_string = self.quality_report()
            file.write(quality_report_string)

            # Write diagnostic report string to file
            diagnostic_report_string = self.diagnostic_report()
            file.write(diagnostic_report_string)

    def one_column_boundary(self, column_name):
        boundary = BoundaryAdherence.compute(
            real_data=self.real_data[column_name],
            synthetic_data=self.synthetic_data[column_name])

        print(boundary)

    def plot_one_column_boundary(self, column_name):
        fig = utils.get_column_plot(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            column_name=column_name,
            metadata=self.metadata.to_dict()
        )

        fig.update_layout(title=column_name)
        fig.show()

    def plot_all_columns_boundaries(self, columns):
        n_cols = len(columns)
        n_rows = (n_cols - 1) // 2 + 1  # 2 columns per row

        # create subplot layout
        fig = sp.make_subplots(rows=n_rows, cols=2, subplot_titles=columns)
        for i, column_idx in enumerate(columns):
            row = i // 2 + 1
            col = i % 2 + 1
            plotly_fig = utils.get_column_plot(
                real_data=self.real_data,
                synthetic_data=self.synthetic_data,
                column_name=column_idx,
                metadata=self.metadata.to_dict()
            )
            for trace in plotly_fig.data:
                fig.add_trace(trace, row=row, col=col)

        fig.update_layout(height=800, width=800, showlegend=False)
        fig.write_image(self.result_path + "all_boundaries_plot.png")




cd
cd GCNReportGenNoEncoder
/opt/conda/envs/MIRQI/bin/python evaluate1.py --reports_path_gt val_gts.csv --output_path val_gts_labeled.csv
/opt/conda/envs/MIRQI/bin/python evaluate1.py --reports_path_gt val_res.csv --output_path val_res_labeled.csv
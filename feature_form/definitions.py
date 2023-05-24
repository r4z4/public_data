import featureform as ff
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ff.register_user("r4z4").make_default_owner()

local = ff.register_local()

applications = local.register_file(
    name="5_8_weekly_loc_data_applications",
    variant="quickstart",
    description="A dataset of applications for the week of 5/8/23",
    path="5_8_weekly_loc_data.parquet"
)

# define a DF transformation on our dataset.
@local.df_transformation(variant="quickstart",
                         inputs=[("applications", "quickstart")])
def average_location_decision_score(applications):
    """the average application decision score """
    return applications.groupby("location_name")["decision_score"].mean()

@ff.entity
class User:
    avg_decision_scores = ff.Feature(
        average_location_decision_score[["location_name", "decision_score"]],
        variant="quickstart",
        type=ff.Int32,
        inference_store=local,
    )

    fraudulent = ff.Label(
        applications[["location_name", "IsFraud"]],
        variant="quickstart",
        type=ff.Bool,
    )

# ff.Labels & ff.Features above
ff.register_training_set(
    "fraud_training", "quickstart",
    label=("fraudulent", "quickstart"),
    features=[("avg_decision_scores", "quickstart")],
)
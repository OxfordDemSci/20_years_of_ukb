# Standard library imports
import ast
import io
import json
import logging
import os
import re
import time
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

# Third-party imports
import dimcli
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def login_dimcli():
    def get_apikey(key_file_path):
        if not key_file_path.is_file():
            raise FileNotFoundError(f"API key file not found at {key_file_path.resolve()}")
        with key_file_path.open("r") as file:
            api_key = file.read().strip()
        return api_key

    dimcli.login(key=get_apikey(Path("../keys/apikey.txt")),
                 endpoint="https://app.dimensions.ai/api/dsl/v2")
    dsl = dimcli.Dsl()
    return dsl


def remove_html_tags(text):
    if pd.isna(text):
        return text
    return re.sub(r'<[^>]*>', '', text)


def get_ukb_showcase_data(timestamp):
    url = "https://biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=19"
    destination_dir = "../data/raw"
    os.makedirs(destination_dir, exist_ok=True)
    filename = f"ukb_data_{timestamp}.txt"
    if os.path.exists(os.path.join(destination_dir, filename)) is False:
        response = requests.get(url)
        response.raise_for_status()
        with open(os.path.join(destination_dir, filename), "wb") as f:
            f.write(response.content)
        print(f"Downloaded and saved: {os.path.join(destination_dir, filename)}")
    df = pd.read_csv(os.path.join(destination_dir, filename), sep='\t')
    return wrangle_raw(df)


def wrangle_raw(df):
    print('Number of records which we start with: ', len(df))
    destination_dir = "../data/raw"

    df = df.rename({'pubmed_id': 'pmid'}, axis=1)
    df['doi'] = df['doi'].str.lower()
    df['title'] = df['title'].str.upper()
    df['title'] = df['title'].apply(remove_html_tags)

    df_doi_nn = df[df['doi'].notnull()]
    dups = df_doi_nn[df_doi_nn.duplicated(subset=['doi'], keep=False)]
    dups = dups.sort_values(by='doi')
    print('Number of duplicated DOIS (excl. NaN): ', len(dups))
    if len(dups) > 0:
        filename = f"ukb_data_duplicated_DOIs.txt"
        print(f"Duplicate DOIs saved: {os.path.join(destination_dir, filename)}")
        print('Note: looks like one DOI occurs 3x...')
        dups.to_csv(os.path.join(destination_dir, filename))
        df_nonnan = df[df['doi'].notna()]
        df_nan = df[df['doi'].isna()]
        df = pd.concat([
            df_nonnan.drop_duplicates(subset=['doi'], keep='first'),
            df_nan], ignore_index=True)
        print(f'Dropping all but the first DOIs ({len(dups) - len(dups['doi'].unique())}) records.')
    df_pmid_nn = df[(df['pmid'].notnull()) & (df['pmid'] != 0)]
    dups = df_pmid_nn[df_pmid_nn.duplicated(subset=['pmid'], keep=False)]
    dups = dups.sort_values(by='pmid')
    print('Number of duplicated pmid (excl. NaN and pmid==0): ', len(dups))
    if len(dups) > 0:
        filename = f"ukb_data_duplicated_pmid.txt"
        print(f"Duplicate pmid saved: {os.path.join(destination_dir, filename)}")
        dups.to_csv(os.path.join(destination_dir, filename))
        df_nonnan = df[df['pmid'].notna()]
        df_nan = df[df['pmid'].isna()]
        df = pd.concat([
            df_nonnan.drop_duplicates(subset=['pmid'], keep='first'),
            df_nan], ignore_index=True)
        print('Dropping all but the first pmid.')

    dups = df[df.duplicated(subset=['title'], keep=False)]
    dups = dups.sort_values(by='title')
    print('Number of duplicated titles (inc. NaN): ', len(dups))
    if len(dups) > 0:
        filename = f"ukb_data_duplicated_titles.txt"
        print(f"Duplicate titles saved: {os.path.join(destination_dir, filename)}")
        dups.to_csv(os.path.join(destination_dir, filename))
        print('Note: Not dropping (unique DOIs which should be retreivable): curious, though...')
    #       df = df.drop_duplicates(subset=['title'], keep='first')

    dups = df[df.duplicated(subset=['pub_id'], keep=False)]
    dups = dups.sort_values(by='pub_id')
    print('Number of duplicated pub_id (inc. NaN): ', len(dups))
    if len(dups) > 0:
        filename = f"ukb_data_duplicated_pub_id.txt"
        print(f"Duplicate pub_id saved: {os.path.join(destination_dir, filename)}")
        dups.to_csv(os.path.join(destination_dir, filename))
        print('Dropping all but the first-seen pub_id.')
        df = df.drop_duplicates(subset=['pub_id'], keep='first')
    print('Number of NaNs in DOI column: ', len(df[df['doi'].isnull()]))
    if len(df[df['doi'].isnull()]) > 0:
        filename = f"ukb_data_doi_nan.txt"
        print(f'These NaNs are saved out to:, {os.path.join(destination_dir, filename)}')
        print('Note: At least some of these NaNs _should_ have DOIs, though...')
        df[df['doi'].isnull()].to_csv(os.path.join(destination_dir, filename))
    print('Number of records with no doi or pmid: ',
          len(df[(df['doi'].isnull()) & (df['pmid'].isnull())]))
    print('Number of records which we are left with: ', len(df))
    return df


def get_pubs(string_representation, field, limit, logger=None):
    query = f"""search publications
    where {field} in {string_representation}
    return publications[abstract + acknowledgements + altmetric + altmetric_id +
                         authors + authors_count + book_doi + book_title +
                         category_bra + category_for + category_for_2020 +
                         category_hra + category_sdg + category_uoa + 
                         clinical_trial_ids + concepts + concepts_scores +
                         date + date_inserted + date_online + date_print +
                         dimensions_url + document_type + doi + editors +
                         field_citation_ratio + funder_countries + funders +
                         funding_section + id + isbn + issn + issue + journal + 
                         journal_lists + journal_title_raw + linkout + mesh_terms +
                         open_access + pages + pmcid + pmid + proceedings_title +
                         publisher + recent_citations + reference_ids +
                         referenced_pubs + relative_citation_ratio + research_org_cities +
                         research_org_countries + research_org_country_names +
                         research_org_names + research_org_state_codes +
                         research_org_state_names + research_org_types +
                         research_orgs + researchers + resulting_publication_doi +
                         score + source_title + subtitles + supporting_grant_ids +
                         times_cited + title + type + volume + year] limit {limit}"""
    capture = io.StringIO()
    with redirect_stdout(capture), redirect_stderr(capture):
        result = dsl.query(query)
    captured_output = capture.getvalue()
    if logger is not None:
        logger.info("Captured DSL query output: %s", captured_output)
    return result.as_dataframe()


def get_raw_data(limit, fields, df, timestamp):
    max_retries = 100
    for field in fields:
        field_log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"../logging/dimensions/api/{field}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        logfile = f"{log_dir}/logfile_{field_log_time}.txt"
        logger = logging.getLogger(field)
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.FileHandler(logfile)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info(f"Starting work on the {field} field")
        if field == 'doi':
            data_to_get = df[df['doi'].notnull()]['doi'].tolist()
        elif field == 'pmid':
            data_to_get = df[df['doi'].isnull()]['pmid'].tolist()
        elif field =='id':
            data_to_get = df['target_id'].unique()
        else:
            print('Some weird field has been passed in')
            break
        for i in tqdm(range(0, len(data_to_get), limit), desc=f"Processing {field} chunks"):
            fpath = f'../data/dimensions/api/raw/{field}/{timestamp}/df_{i}_to_{i + limit}.csv'
            if not os.path.exists(fpath):
                directory = os.path.dirname(fpath)
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                logger.info(f"Processing file: {fpath}")
                chunk = [int(x) if isinstance(x, np.integer) else x for x in data_to_get[i:i + limit]]
                string_representation = json.dumps(chunk)
                for attempt in range(max_retries):
                    try:
                        df_chunk = get_pubs(string_representation, field, limit, logger)
                        break
                    except Exception as e:
                        logger.error("Error encountered for chunk %d (attempt %d/%d): %s",
                                     i, attempt+1, max_retries, e)
                        if attempt < max_retries - 1:
                            sleep_time = 2 ** attempt  # Exponential backoff.
                            logger.info("Retrying in %d seconds...", sleep_time)
                            time.sleep(sleep_time)
                        else:
                            logger.info("Max retries reached. Skipping this chunk.")
                            print('Warning! A chunk couldnt be collected at all?! INVESTIGATE!')
                            df_chunk = None
                if df_chunk is not None:
                    df_chunk.to_csv(fpath)
    logger.info("Tada!")


def merger(directory) -> pd.DataFrame:
    csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path, index_col=0)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def choose_preferred_row(group):
    if group['reference_ids'].notna().any():
        return group[group['reference_ids'].notna()].iloc[[0]]
    else:
        return group.iloc[[0]]


def evaluate_raw_scrape(df, timestamp):
    doi_raw = merger(f'../data/dimensions/api/raw/doi/{timestamp}')
    pmid = merger(f'../data/dimensions/api/raw/pmid/{timestamp}')
    eval_path = f'../data/dimensions/api/raw/eval/{timestamp}'

    print('We start off expecting to get back this number of rows: ', len(df))
    print('We expect to get this many doi: ', len(df[df['doi'].notnull()]['doi'].unique()))
    print('We expect to get this many pmid: ', len(df[df['doi'].isnull()]['pmid'].unique()))
    print('We actually get back this many dois: ', len(doi_raw))
    print('We actually get back this unique dois: ', len(doi_raw['doi'].unique()))

    dups = doi_raw[doi_raw.duplicated(subset=['doi'], keep=False)]
    print(f'Save out the {len(dups)}/2 duplicates to ', os.path.join(eval_path, 'doi_scrape_duplicates.csv'))
    dups.to_csv(os.path.join(eval_path, 'doi_scrape_duplicates.csv'))
    print('If a DOI is duplicated, keep the one which has reference_ids (or the first one seen).')
    doi_cleaned = (
        doi_raw
        .assign(_has_refs=doi_raw['reference_ids'].notna())  # mark preferred rows
        .sort_values(by='_has_refs', ascending=False)  # sort to make them come first
        .drop('_has_refs', axis=1)
        .drop_duplicates(subset='doi', keep='first')  # keep first (preferred)
        .reset_index(drop=True)
    )
    print('We now have this many dois: ', len(doi_cleaned))
    no_doi = df[df['doi'].notnull()][~df[df['doi'].notnull()]['doi'].isin(doi_cleaned['doi'])]
    nodoi_path = os.path.join(eval_path, 'doi_not_in_dim.tsv')
    no_doi.to_csv(nodoi_path, sep='\t')
    print(f"The {len(no_doi)} DOIs not returned are saved at {nodoi_path}")
    print("Note: some of these are clearly non-indexed preprints.")
    print("Note: Some of are on dimensions.ai app?'")
    print("Note: Some of arent on dimensions.ai app, but look like they should be? e.g. 10.1038/s41588-018-0147-3")
    print('We get this many from the pmid search: ', len(pmid['id'].unique()))
    if set(pmid['pmid'].tolist()) == set(df[df['doi'].isnull()]['pmid'].tolist()):
        print('Cool, looks like we got all the pmids without issue')
    else:
        print('?? not currently a problem')

    df_dim = pd.concat([doi_cleaned, pmid], axis=0, ignore_index=True)
    print('We got this many rows of data from Dimensions: ', len(df_dim))
    print('Note: different to len(df) from i.) drop duplicates in wrangle_raw(), ii.) non-returns (sum to diff)')
    return doi_cleaned, pmid, df_dim


def make_long_refs(df_dim):
    def safe_parse(x):
        if isinstance(x, list):
            return x
        elif pd.isna(x):
            return []
        elif isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []  # or raise, depending on how strict you want to be
        else:
            return []

    df_dim['reference_ids'] = df_dim['reference_ids'].apply(safe_parse)
    num_empty_lists = df_dim['reference_ids'].apply(lambda x: x == []).sum()
    print(f"Number of rows where reference_ids is an empty list: {num_empty_lists}")
    df_exploded = df_dim[['id', 'reference_ids']].explode('reference_ids', ignore_index=True)
    print(f'{df_exploded[['id', 'reference_ids']].duplicated().sum()} duplicates in the id-reference_id pair.')
    print('Drop the NaN exploded reference ids (empty lists)')
    df_exploded = df_exploded[df_exploded['reference_ids'].notnull()]
    print(f"Number of rows with NaN in either source_id or target_id: {df_exploded.isna().any(axis=1).sum()}")
    df_exploded = df_exploded.rename(columns={'id': 'source_id', 'reference_ids': 'target_id'})
    print(f'We have {len(df_exploded)} source:target id pairs,'
          f'but only {len(df_exploded['target_id'].unique())} refs to get')
    return df_dim, df_exploded


def save_combined(fpath, doi, pmid, df_dim, refs, df_exploded):
    doi.to_csv(os.path.join(fpath, 'doi.tsv'), sep='\t')
    pmid.to_csv(os.path.join(fpath, 'pmid.tsv'), sep='\t')
    df_dim.to_csv(os.path.join(fpath, 'df_dim.tsv'), sep='\t')
    refs.to_csv(os.path.join(fpath, 'refs.tsv'), sep='\t')
    df_exploded.to_csv(os.path.join(fpath, 'df_exploded.tsv'), sep='\t')
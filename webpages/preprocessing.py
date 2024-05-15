import pandas as pd
import numpy as np
from numpy import sqrt
from math import sin, cos, atan2, radians
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import datetime as dt



def print_metrics(model, X_data, y_data, retype_pred=False, ptitle="Matice záměn"):
    predictions = model.predict(X_data)
    if retype_pred:
        predictions = predictions.astype("str")
    ba = balanced_accuracy_score(y_data.astype("str"), predictions)
    f1_micro = f1_score(y_data.astype("str"), predictions, average="micro")
    f1_macro = f1_score(y_data.astype("str"), predictions, average="macro")
    ra = roc_auc_score(y_data.astype("str"), model.predict_proba(X_data)[:, 1])
    print(f"Balanced accuracy: {ba}")
    print(f"F1 score micro: {f1_micro}")
    print(f"F1 score macro: {f1_macro}")
    print(f"RocAuc score: {ra}")
    print("Confusion matrix:")
    ConfusionMatrixDisplay(confusion_matrix(y_data.astype("str"), predictions)).plot(cmap='Blues')
    plt.xlabel('Predikce modelu')
    plt.ylabel('Skutečnost')
    plt.title(ptitle)
    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.grid(False)
    plt.show()


def get_distance(id_company, id_contracter, address):
    company_address = address.loc[address['id'] == id_company]
    contracter_address = address.loc[address['id'] == id_contracter]
    try:
        lat1 = radians(company_address.latitude.iloc[0])
        lat2 = radians(contracter_address.latitude.iloc[0])
        lon1 = radians(company_address.longitude.iloc[0])
        lon2 = radians(contracter_address.longitude.iloc[0])
        R = 6371
        diff_lat = abs(lat1 - lat2)
        diff_log = abs(lon1 - lon2)
        a = sin(diff_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(diff_log / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    except:
        return np.NAN


class Preprocessor:
    def __init__(self, file_name):
        self.file_name = file_name
        self.dataframes = self.get_dataframes()
        self.address = self.dataframes['address']
        self.company = self.dataframes['company']
        self.contact_person = self.dataframes['contact_person']
        self.contracting_authority = self.dataframes['contracting_authority']
        self.offer = self.dataframes['offer']
        self.procurement = self.dataframes['procurement']

    def __set_intervals(self) -> pd.DataFrame:
        procurement = self.procurement.copy()
        procurement["bids_submission_deadline"] = pd.to_datetime(procurement["bids_submission_deadline"],
                                                                 format='%Y-%m-%d', errors='coerce')
        procurement["date_of_contract_close"] = pd.to_datetime(procurement["date_of_contract_close"], format='%Y-%m-%d',
                                                               errors='coerce')
        procurement["date_of_publication"] = pd.to_datetime(procurement["date_of_publication"], format='%Y-%m-%d',
                                                            errors='coerce')
        procurement['publication_close_interval'] = (
                procurement['date_of_contract_close'] - procurement['date_of_publication']).dt.days
        procurement['bids_close_interval'] = (
                procurement['bids_submission_deadline'] - procurement['date_of_publication']).dt.days
        return procurement

    def get_dataframes(self) -> dict:
        """
        This function reads the csv files and returns a dictionary of dataframes
        :param file_name:
        :return: dictionary of dataframes
        """
        address = pd.read_csv(self.file_name + '/address.csv', low_memory=False)
        company = pd.read_csv(self.file_name + '/company.csv', low_memory=False)
        contact_person = pd.read_csv(self.file_name + '/contact_person.csv', low_memory=False)
        contracting_authority = pd.read_csv(self.file_name + '/contracting_authority.csv', low_memory=False)
        offer = pd.read_csv(self.file_name + '/offer.csv', low_memory=False)
        procurement = pd.read_csv(self.file_name + '/procurement.csv', low_memory=False)

        return {'address': address, 'company': company, 'contact_person': contact_person,
                'contracting_authority': contracting_authority, 'offer': offer, 'procurement': procurement}

    def get_predictions_dataframe(self) -> pd.DataFrame:
        """
        This function preproccesses the data and returns a train, validation and test dataframes
        :param dataframes: dictionary of dataframes
        :return: dataframe of predictions
        """
        offer = self.dataframes['offer']
        procurement = self.__set_intervals()
        company = self.dataframes['company']
        contracting_authority = self.dataframes['contracting_authority']
        address = self.dataframes['address']
        procurement.drop(columns=['is_association_of_suppliers'], inplace=True)
        # get the number of offers for each procurement
        num_of_offers = (offer['procurement_id'].value_counts()).to_frame()
        num_of_offers.rename(columns={"count": "number_of_offers"}, inplace=True)
        # merge the number of offers with the offer dataframe
        offer_with_counts = pd.merge(offer, num_of_offers, on='procurement_id', how='outer')
        df_for_pred = pd.merge(offer_with_counts, procurement, left_on='procurement_id', right_on='id', how='left')
        df_for_pred['is_winner'] = (df_for_pred['company_id'] == df_for_pred['supplier_id'])
        df_for_pred.drop(columns=['id_y', 'procurement_id'], inplace=True)
        df_for_pred = pd.merge(df_for_pred, company, left_on='company_id', right_on='id', how='left')
        # get rid of unnecessary columns
        df_for_pred.drop(columns=['id'], inplace=True)
        df_for_pred.drop(
            columns=['supplier_id', 'id_x', 'procurement_name', 'name_from_nipez_codelist', 'system_number',
                     'vat_id_number', 'company_name'], inplace=True)
        df_for_pred = pd.merge(df_for_pred, contracting_authority, left_on='contracting_authority_id', right_on='id',
                               how='left')
        df_for_pred = df_for_pred[df_for_pred['date_of_publication'].notna()]
        # get rid of duplicates
        df_for_pred.sort_values(by='date_of_publication', ascending=True, inplace=True)
        df_for_pred.drop_duplicates(inplace=True)
        # splitting data
        train_data = df_for_pred.drop(columns=["is_winner"])
        train_labels = df_for_pred["is_winner"]

        X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.30, shuffle=False)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, shuffle=False)

        total_wins = X_train.copy()
        total_wins['is_winner'] = y_train
        total_wins = total_wins.groupby('company_id')['is_winner'].sum().to_frame()
        total_wins.rename(columns={'is_winner': 'total_wins'}, inplace=True)
        total_offers = (X_train['company_id'].value_counts()).to_frame()
        total_offers.rename(columns={'count': 'total_offers'}, inplace=True)
        total_procurements = (X_train['contracting_authority_id'].value_counts()).to_frame()
        total_procurements.rename(columns={'count': 'total_procurements'}, inplace=True)

        X_train = pd.merge(X_train, total_offers, left_on='company_id', right_on='company_id', how='left')
        X_test = pd.merge(X_test, total_offers, left_on='company_id', right_on='company_id', how='left')
        X_val = pd.merge(X_val, total_offers, left_on='company_id', right_on='company_id', how='left')
        X_train = pd.merge(X_train, total_wins, left_on='company_id', right_on='company_id', how='left')
        X_test = pd.merge(X_test, total_wins, left_on='company_id', right_on='company_id', how='left')
        X_val = pd.merge(X_val, total_wins, left_on='company_id', right_on='company_id', how='left')
        X_train = pd.merge(X_train, total_procurements, left_on='contracting_authority_id',
                           right_on='contracting_authority_id', how='left')
        X_test = pd.merge(X_test, total_procurements, left_on='contracting_authority_id',
                          right_on='contracting_authority_id', how='left')
        X_val = pd.merge(X_val, total_procurements, left_on='contracting_authority_id',
                         right_on='contracting_authority_id', how='left')

        names_to_drop = ['company_id', 'contact_person_id', 'organisation_id', 'contracting_authority_id',
                         'contract_price_vat', 'code_from_nipez_codelist', 'contract_price_vat',
                         'contract_price_with_amendments', 'contract_price_with_amendments_vat', 'price_vat',
                         'contract_price', 'date_of_contract_close', 'date_of_publication', 'bids_submission_deadline',
                         'price', 'address_id_y', 'address_id_x', 'url', 'contracting_authority_name', 'id']

        bins = [0, 15, 50, 100, np.inf]
        names = ['[0,15)', '[15,50)', '[50,100)', '[100,+inf)']
        dfs = X_train, X_test, X_val

        for df in dfs:
            df.loc[:, 'distance'] = df.apply(lambda x: get_distance(x['address_id_x'], x['address_id_y'], address),
                                             axis=1)
            df['num_of_nan'] = df.isna().sum(axis=1)
            df['distance'] = pd.cut(df['distance'], bins, labels=names)
            df['place_of_performance'].replace(
                {'Hlavní město Praha': 'Praha', 'EXTRA-REGIO': 'Extra-Regio', np.nan: 'Unknown'}, inplace=True)
            df['public_contract_regime'].replace({np.nan: 'Unknown'}, inplace=True)
            df['type'].replace({np.nan: 'Unknown'}, inplace=True)
            df['type_of_procedure'].replace({np.nan: 'Unknown'}, inplace=True)
            # na nových datech předpokládáme, že firma se zúčastnila alespoň jedné zakázky
            # a tedy i zadavatel alespoň jednu zakázku uvedl
            df['total_wins'].replace({np.nan: 0}, inplace=True)
            df['total_wins'] = df['total_wins'].astype("int")
            df['total_offers'].replace({np.nan: 1}, inplace=True)
            df['total_offers'] = df['total_offers'].astype("int")
            df['total_procurements'].replace({np.nan: 1}, inplace=True)
            df['total_procurements'] = df['total_procurements'].astype("int")

            df['distance'] = df['distance'].astype("str")
            df['distance'].replace({np.nan: 'Unknown'}, inplace=True)

        X_train_all = X_train.copy()
        X_test_all = X_test.copy()
        X_val_all = X_val.copy()

        for df in dfs:
            df.drop(columns=names_to_drop, inplace=True)

        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_all, X_val_all, X_test_all

    def get_data_for_web(self, date=None) -> pd.DataFrame:
        offer = self.dataframes['offer']
        procurement = self.__set_intervals()
        company = self.dataframes['company']
        contracting_authority = self.dataframes['contracting_authority']
        address = self.dataframes['address']
        procurement.drop(columns=['is_association_of_suppliers'], inplace=True)
        # get the number of offers for each procurement
        num_of_offers = (offer['procurement_id'].value_counts()).to_frame()
        num_of_offers.rename(columns={"count": "number_of_offers"}, inplace=True)
        # merge the number of offers with the offer dataframe
        offer_with_counts = pd.merge(offer, num_of_offers, on='procurement_id', how='outer')
        df_for_pred = pd.merge(offer_with_counts, procurement, left_on='procurement_id', right_on='id', how='left')
        df_for_pred['is_winner'] = (df_for_pred['company_id'] == df_for_pred['supplier_id'])
        df_for_pred.drop(columns=['id_y', 'procurement_id'], inplace=True)
        df_for_pred = pd.merge(df_for_pred, company, left_on='company_id', right_on='id', how='left')
        # get rid of unnecessary columns
        df_for_pred.drop(columns=['id'], inplace=True)
        df_for_pred.drop(
            columns=['supplier_id', 'id_x', 'procurement_name', 'name_from_nipez_codelist', 'system_number',
                     'vat_id_number', 'company_name'], inplace=True)
        df_for_pred = pd.merge(df_for_pred, contracting_authority, left_on='contracting_authority_id', right_on='id',
                               how='left')
        df_for_pred = df_for_pred[df_for_pred['date_of_publication'].notna()]
        # get rid of duplicates
        df_for_pred.sort_values(by='date_of_publication', ascending=True, inplace=True)
        df_for_pred.drop_duplicates(inplace=True)
        # splitting data based on month
        before_date_X = df_for_pred[df_for_pred['date_of_publication'] < date].drop(columns=['is_winner'])
        before_date_y = df_for_pred[df_for_pred['date_of_publication'] < date]['is_winner']
        date_X = df_for_pred[(df_for_pred.date_of_publication.dt.month == date.month) & (
                    df_for_pred.date_of_publication.dt.year == date.year)].drop(columns=['is_winner'])
        date_y = df_for_pred[(df_for_pred.date_of_publication.dt.month == date.month) & (
                    df_for_pred.date_of_publication.dt.year == date.year)]['is_winner']

        total_wins = before_date_X.copy()
        total_wins['is_winner'] = before_date_y
        total_wins = total_wins.groupby('company_id')['is_winner'].sum().to_frame()
        total_wins.rename(columns={'is_winner': 'total_wins'}, inplace=True)
        total_offers = (before_date_X['company_id'].value_counts()).to_frame()
        total_offers.rename(columns={'count': 'total_offers'}, inplace=True)
        total_procurements = (before_date_X['contracting_authority_id'].value_counts()).to_frame()
        total_procurements.rename(columns={'count': 'total_procurements'}, inplace=True)

        before_date_X = pd.merge(before_date_X, total_offers, left_on='company_id', right_on='company_id', how='left')
        date_X = pd.merge(date_X, total_offers, left_on='company_id', right_on='company_id', how='left')
        before_date_X = pd.merge(before_date_X, total_wins, left_on='company_id', right_on='company_id', how='left')
        date_X = pd.merge(date_X, total_wins, left_on='company_id', right_on='company_id', how='left')
        before_date_X = pd.merge(before_date_X, total_procurements, left_on='contracting_authority_id',
                           right_on='contracting_authority_id', how='left')
        date_X = pd.merge(date_X, total_procurements, left_on='contracting_authority_id',
                            right_on='contracting_authority_id', how='left')

        names_to_drop = ['company_id', 'contact_person_id', 'organisation_id', 'contracting_authority_id',
                         'contract_price_vat', 'code_from_nipez_codelist', 'contract_price_vat',
                         'contract_price_with_amendments', 'contract_price_with_amendments_vat', 'price_vat',
                         'contract_price', 'date_of_contract_close', 'date_of_publication', 'bids_submission_deadline',
                         'price', 'address_id_y', 'address_id_x', 'url', 'contracting_authority_name', 'id']

        bins = [0, 15, 50, 100, np.inf]
        names = ['[0,15)', '[15,50)', '[50,100)', '[100,+inf)']
        dfs = date_X, before_date_X

        for df in dfs:
            df.loc[:, 'distance'] = df.apply(lambda x: get_distance(x['address_id_x'], x['address_id_y'], address),
                                             axis=1)
            df['num_of_nan'] = df.isna().sum(axis=1)
            df['distance'] = pd.cut(df['distance'], bins, labels=names)
            df['place_of_performance'].replace(
                {'Hlavní město Praha': 'Praha', 'EXTRA-REGIO': 'Extra-Regio', np.nan: 'Unknown'}, inplace=True)
            df['public_contract_regime'].replace({np.nan: 'Unknown'}, inplace=True)
            df['type'].replace({np.nan: 'Unknown'}, inplace=True)
            df['type_of_procedure'].replace({np.nan: 'Unknown'}, inplace=True)
            df['total_wins'].replace({np.nan: 0}, inplace=True)
            df['total_wins'] = df['total_wins'].astype("int")
            df['total_offers'].replace({np.nan: 1}, inplace=True)
            df['total_offers'] = df['total_offers'].astype("int")
            df['total_procurements'].replace({np.nan: 1}, inplace=True)
            df['total_procurements'] = df['total_procurements'].astype("int")

            df['distance'] = df['distance'].astype("str")
            df['distance'].replace({np.nan: 'Unknown'}, inplace=True)

        before_date_X_all = before_date_X.copy()
        date_X_all = date_X.copy()

        for df in dfs:
            df.drop(columns=names_to_drop, inplace=True)

        return date_X, before_date_X, date_y, before_date_y, date_X_all, before_date_X_all
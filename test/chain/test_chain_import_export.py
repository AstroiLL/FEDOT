import json
import os
import shutil

import pytest

from cases.data.data_utils import get_scoring_case_data_paths
from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData
from utilities.synthetic.chain_template_new import ChainTemplate, extract_subtree_root
from test.chain.test_atomized_model import create_chain_with_several_nested_atomized_model


@pytest.fixture(scope="session", autouse=True)
def creation_model_files_before_after_tests(request):
    # Check for exists files if last time tests crash
    delete_json_models_files()

    create_json_models_files()
    request.addfinalizer(delete_json_models_files)


def create_json_models_files():
    """
    Creating JSON's files for test.
    """
    chain = create_chain()
    chain.save_chain("data/test_chain_convert_to_json.json")

    chain_fitted = create_fitted_chain()
    chain_fitted.save_chain("data/test_fitted_chain_convert_to_json.json")

    chain_empty = Chain()
    chain_empty.save_chain("data/test_empty_chain_convert_to_json.json")


def delete_json_models_files():
    """
    Deletes files and folders created during testing.
    """

    folders_name = ["data/atomized_model_1"]

    for folder_path in folders_name:
        dir_path = os.path.abspath(folder_path)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    files_name = ["data/test_fitted_chain_convert_to_json.json",
                  "data/test_empty_chain_convert_to_json.json",
                  "data/test_chain_convert_to_json.json",
                  "data/1.json",
                  "data/2.json",
                  "data/3.json",
                  "data/4.json",
                  "data/5.json",
                  "data/6.json",
                  "data/7.json",
                  "data/8.json",
                  "data/atomized_model_1.json"]

    for file in files_name:
        full_path = os.path.abspath(file)

        if os.path.isfile(full_path):
            with open(full_path, 'r') as json_file:
                chain_fitted_object = json.load(json_file)

            delete_fitted_models(chain_fitted_object)
            os.remove(full_path)


def delete_fitted_models(chain):
    """
    Delete directory and chain's local fitted models.

    :param chain: chain which model's need to delete
    """
    if chain['nodes']:
        root_node = chain['nodes'][0]
        if 'trained_model_path' in root_node:
            model_path = root_node['trained_model_path']
            if model_path is not None and os.path.exists(model_path):
                dir_path = os.path.dirname(os.path.abspath(model_path))
                shutil.rmtree(dir_path)


def create_chain() -> Chain:
    chain = Chain()
    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_xgboost = PrimaryNode('xgboost')

    node_knn = PrimaryNode('knn')
    node_knn.custom_params = {'n_neighbors': 9}

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_lda, node_knn]

    node_logit_second = SecondaryNode('logit')
    node_logit_second.nodes_from = [node_xgboost, node_lda]

    node_lda_second = SecondaryNode('lda')
    node_lda_second.custom_params = {'n_components': 1}
    node_lda_second.nodes_from = [node_logit_second, node_knn_second, node_logit]

    node_xgboost_second = SecondaryNode('xgboost')
    node_xgboost_second.nodes_from = [node_logit, node_logit_second, node_knn]

    node_knn_third = SecondaryNode('knn')
    node_knn_third.custom_params = {'n_neighbors': 8}
    node_knn_third.nodes_from = [node_lda_second, node_xgboost_second]

    chain.add_node(node_knn_third)

    return chain


def create_fitted_chain() -> Chain:
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    chain = create_chain()
    chain.fit(train_data)

    return chain


def create_four_depth_chain():
    knn_node = PrimaryNode('knn')
    lda_node = PrimaryNode('lda')
    xgb_node = PrimaryNode('xgboost')
    logit_node = PrimaryNode('logit')

    logit_node_second = SecondaryNode('logit', nodes_from=[knn_node, lda_node])
    xgb_node_second = SecondaryNode('xgboost', nodes_from=[logit_node])

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node_second, xgb_node])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    chain = Chain()
    chain.add_node(knn_root)

    return chain


def test_export_chain_to_json_correctly():
    chain = create_chain()
    json_actual = chain.save_chain("data/1.json")

    with open("data/test_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    assert json_actual == json.dumps(json_expected)


def test_chain_template_to_json_correctly():
    chain = create_chain()
    chain_template = ChainTemplate(chain)
    json_actual = chain_template.convert_to_dict()

    with open("data/test_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_import_json_to_chain_correctly():
    chain = Chain()
    chain.load_chain("data/test_chain_convert_to_json.json")
    json_actual = chain.save_chain("data/2.json")

    chain_expected = create_chain()
    json_expected = chain_expected.save_chain("data/3.json")

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_import_json_template_to_chain_correctly():
    chain = Chain()
    chain_template = ChainTemplate(chain)
    chain_template.import_from_json("data/test_chain_convert_to_json.json")
    json_actual = chain_template.convert_to_dict()

    chain_expected = create_chain()
    chain_expected_template = ChainTemplate(chain_expected)
    json_expected = chain_expected_template.convert_to_dict()

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_import_json_to_fitted_chain_correctly():
    chain = Chain()
    chain.load_chain("data/test_fitted_chain_convert_to_json.json")
    json_actual = chain.save_chain("data/4.json")

    with open("data/test_fitted_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    assert json_actual == json.dumps(json_expected)


def test_import_json_to_fitted_chain_template_correctly():
    chain = Chain()
    chain_template = ChainTemplate(chain)
    chain_template.import_from_json("data/test_fitted_chain_convert_to_json.json")
    json_actual = chain_template.convert_to_dict()

    with open("data/test_fitted_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_empty_chain_to_json_correctly():
    chain = Chain()
    chain_template = ChainTemplate(chain)
    json_actual = chain_template.convert_to_dict()

    with open("data/test_empty_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_export_import_for_one_chain_object_correctly():
    """
    This test checks whether it is possible to upload a new model to the same object. In other words,
     apply a sequence of commands to the chain object:
    - load_chain(path_first)
    - load_chain(path_second)
    - load_chain(path_third)
    and the last command will rewrite the chain object correctly.
    """
    chain_fitted = create_fitted_chain()
    json_actual = chain_fitted.save_chain("data/5.json")

    chain_fitted_after = create_chain()
    chain_fitted_after.save_chain("data/6.json")
    chain_fitted_after.load_chain("data/5.json")

    json_expected = chain_fitted_after.save_chain("data/7.json")

    assert json_actual == json_expected


def test_import_custom_json_object_to_chain_and_fit_correctly_no_exception():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    data_path = str(os.path.dirname(__file__))
    json_file_path = os.path.join(data_path, "data", "test_custom_json_template.json")

    chain = Chain()
    chain_template = ChainTemplate(chain)
    chain_template.import_from_json(json_file_path)

    chain.fit(train_data)
    chain.save_chain("data/8.json")


def test_extract_subtree_root():
    chain = create_four_depth_chain()
    chain_template = ChainTemplate(chain)

    expected_types = ['knn', 'logit', 'knn', 'lda', 'xgboost']
    new_root_node_id = 4

    root_node = extract_subtree_root(root_model_id=new_root_node_id,
                                     chain_template=chain_template)

    sub_chain = Chain()
    sub_chain.add_node(root_node)
    actual_types = [node.model.model_type for node in sub_chain.nodes]

    assertion_list = [True if expected_types[index] == actual_types[index] else False
                      for index in range(len(expected_types))]
    assert all(assertion_list)


def test_atomized_chain_import_export_correctly():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    chain = create_chain_with_several_nested_atomized_model()
    chain.fit(train_data)
    json_expected = chain.save_chain("data/atomized_model_1.json")

    chain = Chain()
    chain.load_chain("data/atomized_model_1.json")

    with open("data/atomized_model_1.json", 'r') as json_file:
        json_to_del = json.load(json_file)
        delete_fitted_models(json_to_del)

        dir_path = os.path.abspath("data/atomized_model_1")
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        os.remove("data/atomized_model_1.json")

    json_actual = chain.save_chain("data/atomized_model_1.json")

    assert json_actual == json_expected

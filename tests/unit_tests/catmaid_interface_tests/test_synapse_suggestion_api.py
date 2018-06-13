import pytest

from tests.context import skeleton_synapses

from skeleton_synapses.catmaid_interface import CatmaidSynapseSuggestionAPI

from tests.fixtures import catmaid_mock, compact_detail, synapses_near_skeleton


STACK_ID = 1


@pytest.fixture
def syn_sug_api(catmaid_mock):
    return CatmaidSynapseSuggestionAPI(catmaid_mock, STACK_ID)


@pytest.mark.parametrize('skids,calls', [
    ('1234', 1),
    (1234, 1),
    ([1234, 5678], 2)
])
def test_get_node_infos(syn_sug_api, skids, calls, compact_detail):
    """

    Parameters
    ----------
    syn_sug_api : CatmaidSynapseSuggestionAPI
    """
    syn_sug_api._catmaid.fetch.return_value = compact_detail

    response = syn_sug_api.get_node_infos(skids)
    assert syn_sug_api._catmaid.fetch.call_count == calls
    assert len(response) == len(compact_detail[0]) * calls


@pytest.mark.parametrize('skids,calls', [
    ('1234', 1),
    (1234, 1),
    ([1234, 5678], 2)
])
def test_get_synapses_near_skeletons(syn_sug_api, skids, calls, synapses_near_skeleton):
    """

    Parameters
    ----------
    syn_sug_api : CatmaidSynapseSuggestionAPI
    skids
    calls
    synapses_near_skeleton
    """
    syn_sug_api._catmaid.fetch.return_value = synapses_near_skeleton

    response = syn_sug_api.get_synapses_near_skeletons(skids)

    assert syn_sug_api._catmaid.fetch.call_count == calls
    assert len(response) == len(synapses_near_skeleton['data']) * calls

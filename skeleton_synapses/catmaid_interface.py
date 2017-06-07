import json
from collections import defaultdict

import numpy as np

from catpy.client import CatmaidClientApplication, make_url, CoordinateTransformer
from catpy.export import ExportWidget

NEUROCEAN_CONSTANTS = {
    'skel_id': 11524047,
    'project_id': 1,
    'image_stack_id': 1
}

DEV_CONSTANTS = {
    'skel_id': 18383770,
    'project_id': 4,
    'image_stack_id': 4
}


def get_consecutive(lst):
    """
    Given an iterable of unique integers, return a list of lists where the elements of the inner lists are all
    consecutive and in ascending order.

    >>> get_consecutive([2, 4, 1, 5])
    [[1, 2], [4, 5]]
    """
    sorted_lst = sorted(lst)
    ret_lst = [[sorted_lst.pop(0)]]
    while sorted_lst:
        if sorted_lst[0] == ret_lst[-1][-1] + 1:
            ret_lst[-1].append(sorted_lst.pop(0))
        else:
            ret_lst.append([sorted_lst.pop(0)])

    return ret_lst


def extend_slices(broken_slices):
    """
    Given a dict whose keys are z-indexes of missing slices (as reported by stack_info), return a slice extension
    schema as required by ilastik's stack description json.

    Note: not sure what the value of the input dict should mean, so an assertion error is thrown if it is not 1.

    >>> extend_slices({'349': 1, '350': 1, '351': 1, '99': 1})
    [[98, [99]], [348, [349, 350]], [352, [351]]]
    """
    assert all(value == 1 for value in broken_slices.values()), 'Not sure what to do with broken_slice values != 1'
    d = defaultdict(list)
    for broken_block in get_consecutive(int(item) for item in broken_slices):
        pre, post = min(broken_block) - 1, max(broken_block) + 1
        idxs = list(broken_block)
        while idxs:
            d[pre].append(idxs.pop(0))
            if idxs:
                d[post].append(idxs.pop())

    return [[key, value] for key, value in sorted(d.items(), key=lambda x: x[0])]


def make_tile_url_template(image_base):
    """
    May not be correct for all bases
    """
    return make_url(image_base, "{z_index}/0/{y_index}_{x_index}.jpg")


class CatmaidSynapseSuggestionAPI(CatmaidClientApplication):
    def __init__(self, catmaid_client):
        super(CatmaidSynapseSuggestionAPI, self).__init__(catmaid_client)
        self.export_widget = ExportWidget(catmaid_client)

    def _get_stack_id(self, stack_id_or_title):
        try:
            return int(stack_id_or_title)
        except ValueError:
            stacks = self.get((self.project_id, 'stacks'))
            for stack in stacks:
                if stack['title'] == stack_id_or_title:
                    return stack['id']
            raise ValueError('Stack {} not found for project with ID {}'.format(repr(stack_id_or_title), self.project_id))

    def get_stack_description(self, stack_id_or_title, include_offset=True, cache_tiles=False):
        """
        Generate sufficient information for ilastik to read images from CATMAID.

        Parameters
        ----------
        stack_id_or_title : int or str
            Integer ID or string title of the image stack in CATMAID
        include_offset : bool, optional
            Whether to include the stack offset from the project. Including the offset makes it easier to align the
            skeleton geometry with the CATMAID images, but not including it makes it easier to align the ilastik and
            CATMAID images for debugging purposes. Defaults to True.
        cache_tiles : bool, optional
            Whether to cache the tiles (makes viewing them for debugging easier)

        Returns
        -------
        dict
            Information required by ilastik for getting images from CATMAID
        """
        stack_info = self.get_stack_info(stack_id_or_title)
        stack_mirror = stack_info['mirrors'][0]

        return {
            "_schema_name": "tiled-volume-description",
            "_schema_version": 1.0,

            "name": stack_info['stitle'],
            "format": stack_mirror['file_extension'],  # works for jpg
            "dtype": "uint8",  # not defined in stack_info
            "bounds_zyx": [stack_info['dimension'][axis] for axis in 'zyx'],

            # skeleton files do not necessarily use the same coordinates as the CATMAID viewer/tiles, there may be an
            # offset, encoded here. May not be correct, please check. Using this offset makes the ilastik and catmaid
            # z-coordinates not line up, but the skeleton file does.
            "view_origin_zyx": [
                -int(stack_info['translation'][axis]/stack_info['resolution'][axis]) * include_offset for axis in 'zyx'
                ],

            "resolution_zyx": [stack_info['resolution'][axis] for axis in 'zyx'],
            "tile_shape_2d_yx": [stack_mirror['tile_height'], stack_mirror['tile_width']],

            "tile_url_format": make_tile_url_template(stack_mirror['image_base']),  # may not be correct for all bases

            "output_axes": "xyz",  # DO NOT TOUCH

            "cache_tiles": cache_tiles,  # useful for debug viewing

            "extend_slices": extend_slices(stack_info['broken_slices'])
        }

    def get_project_title(self, stack_id_or_title):
        stack_info = self.get_stack_info(stack_id_or_title)

        return stack_info['ptitle']

    def _get_user_id(self, user_id_or_name):
        try:
            return int(user_id_or_name)
        except ValueError:
            users = self.get('user-list')
            for user in users:
                if user_id_or_name in [user['login'], user['full_name']]:
                    return user['id']
            raise ValueError('User {} not found.'.format(repr(user_id_or_name)))

    def get_connectors(self, user_id_or_name, date_from, date_to):
        """

        Parameters
        ----------
        user_id_or_name
        date_from : datetime.datetime
        date_to : datetime.datetime

        Returns
        -------
        list
        """
        params = dict()

        if user_id_or_name:
            params['completed_by'] = self._get_user_id(user_id_or_name)
        if date_from:
            params['from'] = date_from.strftime('%Y%m%d')
        if date_to:
            params['to'] = date_to.strftime('%Y%m%d')

        return self.get((self.project_id, 'connector/list/completed'), params)

    def get_stack_info(self, stack_id_or_title):
        stack_id = self._get_stack_id(stack_id_or_title)
        return self.get((self.project_id, 'stack', stack_id, 'info'))

    def get_coord_transformer(self, stack_id_or_title=None):
        if stack_id_or_title is None:
            return CoordinateTransformer()
        else:
            stack_id = self._get_stack_id(stack_id_or_title)
            return CoordinateTransformer.from_catmaid(self._catmaid, stack_id)

    def get_transformed_treenode_and_connector_geometry(self, stack_id_or_title, *skeleton_ids):
        transformer = self.get_coord_transformer(stack_id_or_title)
        data = self.export_widget.get_treenode_and_connector_geometry(*skeleton_ids)

        for skid, skel_data in data.items():
            for treenode_id, treenode_data in skel_data['treenodes'].items():
                treenode_data['location'] = tuple(transformer.project_to_stack_array(treenode_data['location']))

            for connector_id, connector_data in skel_data['connectors'].items():
                connector_data['location'] = tuple(transformer.project_to_stack_array(connector_data['location']))

        return data

    # def get_fastest_stack_mirror(self, stack_info):
    #     speeds = dict()
    #     canary_loc = stack_info['canary_location']
    #     for idx, stack_mirror in enumerate(stack_info['mirrors']):
    #         canary_url = make_tile_url_template(stack_mirror['image_base']).format(
    #             x_index=math.floor(canary_loc['x']/stack_mirror['tile_width']),
    #             y_index=math.floor(canary_loc['y']/stack_mirror['tile_width']),
    #             z_index=canary_loc['z']
    #         )
    #         try:
    #             start_time = time.time()
    #             response = requests.get(canary_url, auth=self.auth_token)
    #             roundtrip = time.time() - start_time
    #             assert response.status_code == 200
    #             speeds[idx] = roundtrip
    #         except Exception:
    #             speeds[idx] = float('inf')
    #
    #     fastest_idx = min(speeds.items(), key=lambda x: x[1])[0]
    #     return stack_info['mirrors'][fastest_idx]

    def get_workflow_id(self, stack_id, detection_hash, tile_size=512):
        params = {'stack_id': stack_id, 'detection_hash': detection_hash, 'tile_size': tile_size}
        return self.get(('synapsesuggestor/synapse-detection/workflow'), params)['workflow_id']

    def get_project_workflow_id(self, workflow_id, association_hash):
        params = {'workflow_id': workflow_id, 'association_hash': association_hash}
        return self.get(
            ('synapsesuggestor/treenode-association', self.project_id, 'workflow'), params
        )['project_workflow_id']

    def get_treenode_locations(self, skeleton_id, stack_id_or_title):
        """

        Parameters
        ----------
        skeleton_id
        stack_id_or_title

        Returns
        -------
        numpy.ndarray
        """
        transformer = self.get_coord_transformer(stack_id_or_title)
        treenodes = self.get((self.project_id, 'skeletons', skeleton_id, 'compact-detail'))[0]
        treenode_locations = np.array(treenodes)[:, 3:6]
        return transformer.project_to_stack_array(treenode_locations)

    def get_detected_tiles(self, workflow_id):
        return self.get('synapsesuggestor/synapse-detection/tiles/detected', {'workflow_id': workflow_id})

    def add_synapse_slices_to_tile(self, workflow_id, synapse_slice_ids, tile_idx):
        data = {
            'workflow_id': workflow_id,
            'synapse_slices': list(synapse_slice_ids),
            'x_idx': tile_idx[0],
            'y_idx': tile_idx[1],
            'z_idx': tile_idx[2]
        }

        return self.post('synapsesuggestor/synapse-detection/tiles/insert-synapse-slices', data)

    def agglomerate_synapses(self, synapse_slice_ids):
        return self.get('synapsesuggestor/synapse-detection/agglomerate', {'synapse_slices': list(synapse_slice_ids)})

    def add_treenode_synapse_association(self, project_workflow_id, associations):
        data = {
            'project_workflow_id': project_workflow_id,
            'associations': [json.dumps(association) for association in associations]
        }
        return self.post(('synapsesuggestor/treenode-association', self.project_id, 'add'), data)

    def get_treenode_synapse_association(self, skeleton_id):
        return self.get(('synapsesuggestor/treenode-association', self.project_id, 'get'), {'skid': skeleton_id})

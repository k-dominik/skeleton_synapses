import requests
import json
from collections import defaultdict
import time
import math

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
    sorted_lst = list(sorted(lst))
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


def make_url(base_url, *args):
    """
    Given any number of URL components, join them as if they were a path regardless of trailing and prepending slashes

    >>> make_url('google.com', 'mail')
    'google.com/mail'

    >>> make_url('google.com/', '/mail')
    'google.com/mail'
    """
    for arg in args:
        joiner = '' if base_url.endswith('/') else '/'
        relative = arg[1:] if arg.startswith('/') else arg
        base_url = requests.compat.urljoin(base_url + joiner, relative)

    return base_url


def make_tile_url_template(image_base):
    """
    May not be correct for all bases
    """
    return make_url(image_base, "{z_index}/0/{y_index}_{x_index}.jpg")


class CatmaidAPI(object):
    """
    See catpy [1]_ for alternative implementation/ inspiration

    .. [1] https://github.com/ceesem/catpy
    """

    def __init__(self, base_url, token, auth_name=None, auth_pass=None, project_id_or_title=None):
        self.base_url = base_url

        self.session = requests.Session()
        self.session.auth = self.CatmaidAuthToken(token, auth_name, auth_pass)

        if project_id_or_title is None:
            self.project_id = None
        else:
            self.set_project(project_id_or_title)

    class CatmaidAuthToken(requests.auth.HTTPBasicAuth):
        def __init__(self, token, auth_name=None, auth_pass=None):
            self.token = token
            super(CatmaidAPI.CatmaidAuthToken, self).__init__(auth_name, auth_pass)

        def __call__(self, r):
            r.headers['X-Authorization'] = 'Token {}'.format(self.token)
            return super(CatmaidAPI.CatmaidAuthToken, self).__call__(r)

    def _make_catmaid_url(self, *args):
        return make_url(self.base_url, *args)

    def _get_project_id(self, project_id_or_title):
        try:
            return int(project_id_or_title)
        except ValueError:
            projects = self.get('projects')
            for project in projects:
                if project_id_or_title == project['title']:
                    return project['id']
            raise ValueError('Project with title {} does not exist'.format(repr(project_id_or_title)))

    def set_project(self, project_id_or_title):
        self.project_id = self._get_project_id(project_id_or_title)

    @classmethod
    def from_json(cls, path, with_project_id=True):
        """
        Return a CatmaidAPI instance with credentials matching those in a JSON file. Should have the properties:

        base_url, token, auth_name, auth_pass

        And optionally

        project_id

        Parameters
        ----------
        path : str
            Path to the JSON credentials file
        with_project_id : bool
            Whether to look for the `project_id` field (it can be set later on the returned CatmaidAPI instance)

        Returns
        -------
        CatmaidAPI
            Authenticated instance of the API
        """
        with open(path) as f:
            credentials = json.load(f)
        return cls(
            credentials['base_url'],
            credentials['token'],
            credentials['auth_name'],
            credentials['auth_pass'],
            credentials['project_id'] if with_project_id else None
        )

    def get(self, relative_url, params=None, raw=False):
        """
        Get data from a running instance of CATMAID.

        Parameters
        ----------
        relative_url: str
            URL to send the request to, relative to the base_url
        params: dict or str, optional
            JSON-like key/value data to be included in the get URL (defaults to empty)
        raw: bool, optional
            Whether to return the response as a string (defaults to returning a dict)

        Returns
        -------
        dict or str
            Data returned from CATMAID: type depends on the 'raw' parameter.
        """
        return self.fetch(relative_url, method='GET', data=params, raw=raw)

    def post(self, relative_url, data=None, raw=False):
        """
        Post data to a running instance of CATMAID.

        Parameters
        ----------
        relative_url: str
            URL to send the request to, relative to the base_url
        data: dict or str, optional
            JSON-like key/value data to be included in the request as a payload (defaults to empty)
        raw: bool, optional
            Whether to return the response as a string (defaults to returning a dict)

        Returns
        -------
        dict or str
            Data returned from CATMAID: type depends on the 'raw' parameter.
        """
        return self.fetch(relative_url, method='POST', data=data, raw=raw)

    def fetch(self, relative_url, method='GET', data=None, raw=False):
        """
        Interact with the CATMAID server in a manner very similar to the javascript CATMAID.fetch API.

        Parameters
        ----------
        relative_url: str
            URL to send the request to, relative to the base_url
        method: {'GET', 'POST'}, optional
            HTTP method to use (the default is 'GET')
        data: dict or str, optional
            JSON-like key/value data to be included in the request as a payload (defaults to empty)
        raw: bool, optional
            Whether to return the response as a string (defaults to returning a dict)

        Returns
        -------
        dict or str
            Data returned from CATMAID: type depends on the 'raw' parameter.
        """
        url = self._make_catmaid_url(relative_url)
        data = data or dict()
        if method.upper() == 'GET':
            response = self.session.get(url, params=data)
        elif method.upper() == 'POST':
            response = self.session.post(url, data=data)
        else:
            raise ValueError('Unknown HTTP method {}'.format(repr(method)))

        return response.json() if not raw else response.text

    def get_treenode_and_connector_geometry(self, skeleton_id, stack_id_or_title=None):
        """
        See CATMAID code [2]_ for original js implementation
        .. [2] http://github.com/catmaid/CATMAID/blob/master/django/applications/catmaid/static/js/widgets/export
        -widget.js#L449
        
        Parameters
        ----------
        skeleton_id : int or str
        stack_id_or_title : int or str
            By default (= None), geometry will be returned in project coordinates. If a stack_id_or_title is given, 
            they will be transformed into stack coordinates for the given stack.
        """

        data = self.get('{}/{}/1/0/compact-skeleton'.format(self.project_id, skeleton_id))

        # if stack id is not given, will return a null transformer which does not change its inputs
        coord_transformer = self.get_coord_transformer(stack_id_or_title)

        skeleton = {
            'treenodes': dict(),
            'connectors': dict()
        }

        for treenode in data[0]:
            skeleton['treenodes'][treenode[0]] = {
                'location': [
                    coord_transformer.project_to_stack_coord(dim, coord) for dim, coord in zip('xyz', treenode[3:6])
                ],
                'parent_id': treenode[1]
            }

        for connector in data[1]:
            if connector[2] not in [0, 1]:
                continue

            conn_id = connector[1]
            if conn_id not in skeleton['connectors']:
                skeleton['connectors'][conn_id] = {
                    'presynaptic_to': [],
                    'postsynaptic_to': []
                }

            skeleton['connectors'][conn_id]['location'] = [
                    coord_transformer.project_to_stack_coord(dim, coord) for dim, coord in zip('xyz', connector[3:6])
                ]
            relation = 'postsynaptic_to' if connector[2] == 1 else 'presynaptic_to'
            skeleton['connectors'][conn_id][relation].append(connector[0])

        return {"skeletons": {str(skeleton_id): skeleton}}

    def _get_stack_id(self, stack_id_or_title):
        try:
            return int(stack_id_or_title)
        except ValueError:
            stacks = self.get('{}/stacks'.format(self.project_id))
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
        stack_info = self._get_stack_info(stack_id_or_title)
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
        stack_info = self._get_stack_info(stack_id_or_title)

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
        params = dict()

        if user_id_or_name:
            params['completed_by'] = self._get_user_id(user_id_or_name)
        if date_from:
            params['from'] = None
        if date_to:
            params['to'] = None

        return self.get('{}/connector/list/completed'.format(self.project_id), params)

    def _get_stack_info(self, stack_id_or_title):
        stack_id = self._get_stack_id(stack_id_or_title)
        return self.get('{}/stack/{}/info'.format(self.project_id, stack_id))

    def get_coord_transformer(self, stack_id_or_title=None):
        if stack_id_or_title is None:
            return NullCoordinateTransformer()
        else:
            stack_info = self._get_stack_info(stack_id_or_title)

            return CoordinateTransformer(stack_info['resolution'], stack_info['translation'])

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


class CoordinateTransformer(object):
    def __init__(self, resolution=None, translation=None):
        """
        Helper class for transforming between stack and project coordinates.
        
        Parameters
        ----------
        resolution : dict
            x, y and z resolution of the stack
        translation : dict
            x, y and z the location of the stack's origin (0, 0, 0) in project space
        """
        self.resolution = resolution if resolution else {dim: 1 for dim in 'xyz'}
        self.translation = translation if translation else {dim: 0 for dim in 'xyz'}

    def project_to_stack_coord(self, dim, project_coord):
        return (project_coord - self.translation[dim]) / self.resolution[dim]

    def project_to_stack(self, project_coords):
        """
        Take a point in project space and transform it into stack space.
        
        Parameters
        ----------
        project_coords : dict
            x, y, and/or z coordinates in project/ real space

        Returns
        -------
        dict
            coordinates transformed into stack/voxel space
        """
        return {dim: self.project_to_stack_coord(dim, proj_coord) for dim, proj_coord in project_coords.items()}

    def stack_to_project_coord(self, dim, stack_coord):
        return stack_coord * self.resolution[dim] + self.translation[dim]

    def stack_to_project(self, stack_coords):
        """
        Take a point in stack space and transform it into project space.
        
        Parameters
        ----------
        stack_coords : dict
            x, y, and/or z coordinates in stack/ voxel space

        Returns
        -------
        dict
            coordinates transformed into project/ real space
        """
        return {dim: self.stack_to_project_coord(dim, stack_coord) for dim, stack_coord in stack_coords.items()}


class NullCoordinateTransformer(CoordinateTransformer):
    def project_to_stack_coord(self, dim, project_coord):
        return project_coord

    def project_to_stack(self, project_coords):
        return project_coords.copy()

    def stack_to_project_coord(self, dim, stack_coord):
        return stack_coord

    def stack_to_project(self, stack_coords):
        return stack_coords.copy()


if __name__ == '__main__':
    import doctest
    doctest.testmod()

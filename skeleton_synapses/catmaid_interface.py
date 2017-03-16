import requests
import json
from collections import defaultdict

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


class CatmaidAPI(object):
    """
    See catpy [1]_ for alternative implementation/ inspiration

    .. [1] https://github.com/ceesem/catpy
    """

    def __init__(self, base_url, token, auth_name=None, auth_pass=None, project_id_or_title=None):
        self.base_url = base_url
        self.auth_token = self.CatmaidAuthToken(token, auth_name, auth_pass)

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
            response = requests.get(url, params=data, auth=self.auth_token)
        elif method.upper() == 'POST':
            response = requests.post(url, data=data, auth=self.auth_token)
        else:
            raise ValueError('Unknown HTTP method {}'.format(repr(method)))

        return response.json() if not raw else response.text

    def get_treenode_and_connector_geometry(self, skeleton_id):
        """
        See CATMAID code [2]_ for original js implementation
        .. [2] http://github.com/catmaid/CATMAID/blob/master/django/applications/catmaid/static/js/widgets/export
        -widget.js#L449
        """

        data = self.get('{}/{}/1/0/compact-skeleton'.format(self.project_id, skeleton_id))

        skeleton = {
            'treenodes': dict(),
            'connectors': dict()
        }

        for treenode in data[0]:
            skeleton['treenodes'][treenode[0]] = {
                'location': treenode[3:6],
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

            skeleton['connectors'][conn_id]['location'] = connector[3:6]
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
        stack_id = self._get_stack_id(stack_id_or_title)

        stack_info = self.get('{}/stack/{}/info'.format(self.project_id, stack_id))
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

            "tile_url_format": make_url(  # probably not correct for all image bases
                stack_mirror['image_base'],
                "{z_index}/0/{y_index}_{x_index}.jpg"
            ),

            "output_axes": "xyz",  # DO NOT TOUCH

            "cache_tiles": cache_tiles,  # useful for debug viewing

            "extend_slices": extend_slices(stack_info['broken_slices'])
        }

    def get_stack_project_translation(self, stack_id_or_title):
        """
        Get the pixel offsets of the project from the stack.

        Parameters
        ----------
        stack_id_or_title : int or str
            Integer ID or string title of the image stack in CATMAID

        Returns
        -------
        dict
            Dict of x, y, z offsets of project from stack, in pixels (i.e. the offsets detailed in the project
            stacks admin panel, divided by the resolution, multiplied by -1)
        """
        stack_id = self._get_stack_id(stack_id_or_title)
        stack_info = self.get('{}/stack/{}/info'.format(self.project_id, stack_id))

        return {axis: -int(stack_info['translation'][axis]/stack_info['resolution'][axis]) for axis in 'zyx'}

    def get_project_title(self, stack_id_or_title):
        stack_id = self._get_stack_id(stack_id_or_title)
        stack_info = self.get('{}/stack/{}/info'.format(self.project_id, stack_id))

        return stack_info['ptitle']


if __name__ == '__main__':
    import doctest
    doctest.testmod()

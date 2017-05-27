from odps import ODPS

access_id = ''
access_key = ''
project = ''
endpoint = ''

o = ODPS(access_id=access_id,
            secret_access_key=access_key,
            project=project,
            endpoint=endpoint
    )
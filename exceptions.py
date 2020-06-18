class BlobError(Exception):
    """ Raised when a blob is a point"""
    def __str__(self):
        return "One of the found blobs is a point or almost one."
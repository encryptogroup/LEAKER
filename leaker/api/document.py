"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""


class Document:
    """A document from a data set. It possesses a unique identifier and a length."""

    __id: str
    __length: int
    __payload: str

    def __init__(self, document_id: str, length: int):
        self.__id = document_id
        self.__length = length

    def id(self):
        return self.__id

    def length(self):
        return self.__length

    def __repr__(self) -> str:
        return f"Document({self.__id}, length={self.__length})"


class InputDocument(Document):
    """
    A special document used in pre-processing to write data to a data set. It contains the content of the document as
    well as the values stored by Document.
    """
    __content: str

    def __init__(self, document_id: str, content: str):
        super(InputDocument, self).__init__(document_id, len(content))

        self.__content = content

    def content(self):
        return self.__content

    def __repr__(self):
        content = self.__content
        if len(content) > 32:
            content = content[:32] + "..."
        return f"InputDocument({self.id()}, length={self.length()}, '{content}')"


class QueryInputDocument(InputDocument):
    """
    A special document used in pre-processing to write data to a query log index. It contains the content of the
    search as well as the userid.
    """
    __user_id: str

    def __init__(self, document_id: str, content: str, user_id: str):
        super(QueryInputDocument, self).__init__(document_id, content)

        self.__user_id = user_id

    def user_id(self):
        return self.__user_id

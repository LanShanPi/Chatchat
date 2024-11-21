from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader
import sys
sys.path.append(r"/home/ubuntu/flockai_new/")
from document_loaders.ocr import get_ocr


class RapidOCRLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def img2text(filepath):
            resp = ""
            ocr = get_ocr()
            result, _ = ocr(filepath)
            if result:
                ocr_result = [line[1] for line in result]
                resp += "\n".join(ocr_result)
            return resp

        text = img2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    loader = RapidOCRLoader(file_path="/home/ubuntu/flockai_new/tests/samples/WechatIMG660.jpg")
    docs = loader.load()
    print(docs)

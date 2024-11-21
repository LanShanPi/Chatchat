from typing import TYPE_CHECKING

from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader

from typing import List
import tqdm


from fastapi import FastAPI,Request
import uvicorn
from typing import Optional

app = FastAPI()

if TYPE_CHECKING:
    try:
        from rapidocr_paddle import RapidOCR
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR


def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr

class RapidOCRIMGLoader(UnstructuredFileLoader):
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


class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def pdf2text(filepath):
            import fitz # pyMuPDF里面的fitz包，不要与pip install fitz混淆
            import numpy as np
            ocr = get_ocr()
            doc = fitz.open(filepath)
            resp = ""

            b_unit = tqdm.tqdm(total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0")
            for i, page in enumerate(doc):

                # 更新描述
                b_unit.set_description("RapidOCRPDFLoader context page index: {}".format(i))
                # 立即显示进度条更新结果
                b_unit.refresh()
                # TODO: 依据文本与图片顺序调整处理方式
                text = page.get_text("")
                resp += text + "\n"

                img_list = page.get_images()
                for img in img_list:
                    pix = fitz.Pixmap(doc, img[0])
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                    result, _ = ocr(img_array)
                    if result:
                        ocr_result = [line[1] for line in result]
                        resp += "\n".join(ocr_result)

                # 更新进度
                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


def get_file_type_by_magic_number(file_path):
    with open(file_path, 'rb') as f:
        file_header = f.read(8)
    if file_header.startswith(b'%PDF'):
        return 'pdf'
    elif file_header.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    elif file_header.startswith(b'\x89PNG'):
        return 'png'
    # elif file_header.startswith(b'GIF87a') or file_header.startswith(b'GIF89a'):
    #     return 'gif'
    else:
        return 'unknown'

@app.post("/ocr/")
async def upload_file(request:Request):
    # {"file_path"}
    data = await request.json()
    file_path = data.get("file_path")
    file_type = get_file_type_by_magic_number(file_path)
    if file_type == "pdf":
        loader = RapidOCRPDFLoader(file_path=file_path)
    elif file_type == "unknown":
        return None
    else:
        loader = RapidOCRIMGLoader(file_path=file_path)
    docs = loader.load()
    return docs
        

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

    

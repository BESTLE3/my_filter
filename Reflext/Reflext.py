import reflex as rx
import cv2
import numpy as np
from PIL import Image

from Reflext import style

class State(rx.State):
    img: list[str] = []
    firstimg = []
    filtered_img: str = ""
    pil_image = []

    async def handle_upload(self, files: list[rx.UploadFile]):
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename

            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

            self.img.append(file.filename)

    # 스케치 버튼 함수
    def sketchbutton(self, img_filename):
        if len(self.pil_image) == 0:
            self.firstimg.append(self.img[0])
            self.img = []
            filtered_img, pil_image = sketchfilter(img_filename)
            self.pil_image = []
            self.pil_image.append(pil_image)
            self.img.append(filtered_img)
        else:
            return
        
    # 카툰 버튼 함수
    def cartoonbutton(self, img_filename):
        if len(self.pil_image) == 0:
            self.firstimg.append(self.img[0])
            self.img = []
            filtered_img, pil_image = cartoonfilter(img_filename)
            self.pil_image = []
            self.pil_image.append(pil_image)
            self.img.append(filtered_img)
        else:
            return
        
    # 뽀샵 버튼 함수
    def poshopbutton(self, img_filename):
        if len(self.pil_image) == 0:
            self.firstimg.append(self.img[0])
            self.img = []
            filtered_img, pil_image = poshopfilter(img_filename)
            self.pil_image = []
            self.pil_image.append(pil_image)
            self.img.append(filtered_img)
        else:
            return
        
    # 그레이 배경 버튼 함수
    def graybgbutton(self, img_filename):
        if len(self.pil_image) == 0:
            self.firstimg.append(self.img[0])
            self.img = []
            filtered_img, pil_image = graybackgoundfilter(img_filename)
            self.pil_image = []
            self.pil_image.append(pil_image)
            self.img.append(filtered_img)
        else:
            return
        
        
    def mainimagebutton(self):
        if len(self.pil_image) == 1:
            self.img.pop()
            self.img.append(self.firstimg[0])
            self.pil_image.pop()
        else:
            return
        
    def changeimagebutton(self):
        self.img = []
        self.pil_image = []
        self.firstimg = []
        self.filtered_img = ""


###############################################
# 스케치 필터
def sketchfilter(img_filename):
    # 파일 경로를 통해 이미지를 로드
    img_path = rx.get_upload_dir() / img_filename
    # PIL 이미지를 열고 numpy 배열로 변환
    img = Image.open(img_path)
    img_array = np.array(img)

    # 이미지를 그레이 스케일로 변환
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # 블러 이미지를 만들고 블러 이미지와 그레이 스케일 이미지를 나눈다.
    blur_img = cv2.GaussianBlur(gray_img, (0, 0), 5)
    output_img = cv2.divide(gray_img, blur_img, scale=255)

    # 필터링된 이미지를 Pillow 이미지로 변환
    pil_image = Image.fromarray(output_img)
    # 저장 경로를 설정하고 이미지를 저장
    output_filename = f"sketch_{img_filename}"
    output_path = rx.get_upload_dir() / output_filename
    pil_image.save(output_path, format="PNG")
    return [output_filename], pil_image
###############################################


###############################################
# 카툰 필터
def cartoonfilter(img_filename):
    img_path = rx.get_upload_dir() / img_filename
    img = Image.open(img_path)
    img_array = np.array(img)

    h, w = img_array.shape[:2]
    img2 = cv2.resize(img_array, (w//2, h//2))

    # 크기 조절한 이미지(img2)를 양방향 필터링으로 에지가 아닌 부분만 블러링
    # 크기 조절한 이미지(img2)를 캐니 에지 검출기로 에지 검출 후 255에서 빼주어 흰 부분과 검은 부분 반전
    # 에지 검출한 이미지를 그레이 스케일 이미지로 변환
    BlurImg = cv2.bilateralFilter(img2, -1, 10, 1)
    EdgeImg = 255 - cv2.Canny(img2, 100, 100)
    EdgeImg = cv2.cvtColor(EdgeImg, cv2.COLOR_GRAY2BGR)

    # 블러링한 이미지와 에지 검출한 이미지를 AND연산으로 결합
    # 처음에 크기를 조절했던 이미지를 원래 크기로 조정
    OutputImg = cv2.bitwise_and(BlurImg, EdgeImg)
    OutputImg = cv2.resize(OutputImg, (w,h), interpolation=cv2.INTER_NEAREST)
    pil_image = Image.fromarray(OutputImg)
    output_filename = f"cartoon_{img_filename}"
    output_path = rx.get_upload_dir() / output_filename
    pil_image.save(output_path, format="PNG")
    return [output_filename], pil_image


# 뽀샵 필터
def poshopfilter(img_filename):
    img_path = rx.get_upload_dir() / img_filename
    img = Image.open(img_path)
    img_array = np.array(img)

    alpha = 1.0

    blurimg = cv2.bilateralFilter(img_array, 5, 75, 75)
    outputimg = np.clip((1+alpha) * blurimg - 128 * alpha, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(outputimg)
    output_filename = f"pohsop_{img_filename}"
    output_path = rx.get_upload_dir() / output_filename
    pil_image.save(output_path, format="PNG")
    return [output_filename], pil_image
###############################################


###############################################
# 배경 회색 필터
def apply_grabcut(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)

    # GrabCut 알고리즘에 필요한 임시 배열 생성
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 그랩컷
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 전경 배경 픽셀 설정
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask2

def create_foreground_background(image, mask):
    # 전경 이미지
    foreground = image * mask[:, :, np.newaxis]

    # 배경 흑백 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    background = background * (1 - mask)[:, :, np.newaxis]

    # 전경과 배경 합성
    result = cv2.add(foreground, background)
    return result

# 이미지 불러오기
def graybackgoundfilter(img_filename):
    img_path = rx.get_upload_dir() / img_filename
    img = Image.open(img_path)
    img_array = np.array(img)

    # 사각형 설정
    rect = (1, 1, img_array.shape[1] , img_array.shape[0])

    # GrabCut 알고리즘 적용
    mask = apply_grabcut(img_array, rect)

    # 배경을 흑백으로 변환하고 전경과 결합
    result = create_foreground_background(img_array, mask)

    pil_image = Image.fromarray(result)
    output_filename = f"GRAY_backgound_{img_filename}"
    output_path = rx.get_upload_dir() / output_filename
    pil_image.save(output_path, format="PNG")
    return [output_filename], pil_image
###############################################




# 상단 네비게이션 바
def navbar():
    return rx.hstack(
        rx.hstack(
            rx.text(
                "FILTER",
                size="7", 
                weight="bold",
                color_scheme="jade",
            ),
        ),
        rx.spacer(),
        rx.dialog.root(
            rx.dialog.trigger(rx.button("도움말")),
            rx.dialog.content(
                rx.dialog.title("도움말"),
                rx.dialog.description(
                    rx.text("≫ 필터가 적용된 상태에서 다른 필터를 선택할때 되돌리기 버튼을 누르고 다른 필터를 선택해야합니다."),
                    rx.text("≫ 원본 이미지 상태에서 다운로드 버튼을 누를 경우 잘못된 파일이 다운로드되니 주의하기시 바랍니다(txt파일이 다운로드됩니다.)"),
                    rx.text("≫ gray background 필터 적용시 시간이 조금 걸립니다.")

                    ),
        rx.flex(
        rx.dialog.close(
            rx.button(
                "닫기",
                size="2",
                ),
                ),
            justify="end",
            ),
            ),
        ),
        position="fixed",
        top="0px",
        background_color="#F5EFE6",
        padding="1em",
        height="4em",
        width="100%",
        z_index="5",
    )

# 메인 페이지
def maintext():
    return rx.vstack(
        rx.heading(
            "FILTER",
        ),
        rx.upload(
            rx.text(
                "여기를 클릭하거나 드래그 앤 드롭으로 이미지를 삽입하세요."
            ),
            id="my_upload",
            on_drop=State.handle_upload(rx.upload_files(upload_id="my_upload")),
            border="1px dotted rgb(107,99,246)",
            padding="5em",
            max_files=1,
            accept={
                "image/png": [".png"],
                "image/jpeg": [".jpg", ".jpeg"],
                "image/gif": [".gif"],
                "image/webp": [".webp"],
                "image/bmp": [".bmp"],
            }
        ),
        rx.box(
            rx.foreach(
                State.img,
                lambda img: rx.vstack(
                    rx.image(src=rx.get_upload_url(img)),
                ),
            ),
        columns="2",
        spacing="1",
        ),
        style=style.mainstyle        
    )

# filterpage의 하단 필터 선택 박스
def filterselect():
    return rx.box(
        rx.flex(
                rx.button(
                    "sketch",
                    style={"font_size" : "19px"},
                    on_click=lambda: State.sketchbutton(State.img[-1]),
                    background="linear-gradient(45deg, #7469B6, #AD88C6)",
                    width="100%",
                    height="75px",
                    border_radius="1.5em",
                    _hover={
                        "opacity": 0.7,
                    },
                ),
                rx.button(
                    "cartoon",
                    style={"font_size" : "19px"},
                    on_click=lambda: State.cartoonbutton(State.img[-1]),
                    background="linear-gradient(45deg, #7469B6, #AD88C6)",
                    width="100%",
                    height="75px",
                    border_radius="1.5em",
                    _hover={
                        "opacity": 0.7,
                    },
                    ),
                rx.button(
                    "poshop",
                    style={"font_size" : "19px"},
                    on_click=lambda: State.poshopbutton(State.img[-1]),
                    background="linear-gradient(45deg, #7469B6, #AD88C6)",
                    width="100%",
                    height="75px",
                    border_radius="1.5em",
                    _hover={
                        "opacity": 0.7,
                    },
                    ),
                rx.button(
                    "gray background",
                    style={"font_size" : "19px"},
                    on_click=lambda: State.graybgbutton(State.img[-1]),
                    background="linear-gradient(45deg, #7469B6, #AD88C6)",
                    width="100%",
                    height="75px",
                    border_radius="1.5em",
                    _hover={
                        "opacity": 0.7,
                    },
                ),
            columns=[2],
            display="block",
            justify="center",
            aling_items="center",
            spacing="5",
            #background_color="red",
            # text_align="center",
            # width="500px",
            # height="100px",
            position="absolute",
            top="0",
            bottom="0",
            left="0",
            right="0",
            margin="auto"
        ),
        size="4",
        columns="5",
        bottom="0px",
        left="0px",
        width="100%",
        height="12em",
        z_index="5",
        background_color="#F5EFE6",
        position="fixed",
        border_radius="10px",
    )

# 이미지 삽입 후 필터 선택 페이지
def filterpage():
    return rx.vstack(
        rx.box(
                rx.box(
                    rx.button(
                        "되돌리기",
                        on_click=lambda: State.mainimagebutton(),
                        border_radius="1em",
                        box_shadow="rgba(151, 65, 252, 0.8) 0 15px 30px -10px",
                        background_image="linear-gradient(144deg,#AF40FF,#5B42F3 50%,#00DDEB)",
                        box_sizing="border-box",
                        color="white",
                        opacity=1,
                        _hover={
                            "opacity": 0.7,
                        },
                    ),
                    rx.alert_dialog.root(
                    rx.alert_dialog.trigger(
                        rx.button(
                            "이미지 바꾸기",
                            border_radius="1em",
                            color_scheme="grass",
                            _hover={
                                "opacity": 0.7,
                            },
                            ),
                            ),
                    rx.alert_dialog.content(
                        rx.alert_dialog.title("주의"),
                        rx.alert_dialog.description(
                            "정말 이미지를 바꾸시겠습니다? 기존 작업물은 사라집니다."
                            ),
                        rx.flex(
                        rx.alert_dialog.cancel(
                            rx.button(
                                "닫기",
                                size="3",
                                    ),
                                ),
                        rx.alert_dialog.action(
                            rx.button(
                                "바꾸기",
                                size="3",
                                on_click=lambda: State.changeimagebutton(),
                                color_scheme="red",
                            ),
                        ),
                        spacing="3",
                        justify="end",
                        ),
                    ),
                    ),
                    rx.button(
                        "다운로드",
                        on_click=rx.download(
                            data=State.pil_image[0],
                            filename=State.img[0]
                            ),
                        background_image="linear-gradient(144deg, red, blue)",
                        border_radius="1em",
                        _hover={
                            "opacity": 0.7,
                        },
                    ),
                    spacing="3",
                    width="100%",
                ),
            rx.box(
                rx.foreach(
                    State.img,
                    lambda img: rx.vstack(
                        rx.image(
                            src=rx.get_upload_url(img),
                            ),
                        rx.text("파일 이름 : ", img),
                    ),
                ),
                width="100%",
                justify="center",
            ),
            filterselect(),
            spacing="5",
        ),
        direction="column",
    )

# 메인 코드
def index():
    """ The main app."""
    return rx.fragment( 
        navbar(),
        rx.cond(
            (State.img.length() == 0),
            rx.box(
                maintext(),
                style=style.container
            ),
            rx.box(
                filterpage(),
                style=style.container
        ),
    ),
    )

app = rx.App(style=style.style)
app.add_page(index, title="FILTER")

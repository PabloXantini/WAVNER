from ..vision.controller import Controller

class HandSynthetizer(Controller):
    def __init__(self):
        super().__init__()
    def update(self, image, hand, ctx):
        if not ctx.debug:
            return
        ctx.d.draw_landmarks(
            image,
            hand,
            ctx.h.HAND_CONNECTIONS,
            ctx.d.DrawingSpec(color=(255,0,0), thickness=4, circle_radius=5),
            ctx.d.DrawingSpec(color=(0,0,255), thickness=4)
        )
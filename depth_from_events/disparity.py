from depth_from_events.depth_disparity import DepthDisparityToFlow


class DisparityToFlow(DepthDisparityToFlow):
    def __init__(self, *args, **kwargs):
        super().__init__("depth", *args, **kwargs)

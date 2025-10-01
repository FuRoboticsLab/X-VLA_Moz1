
from xvla import build_xvla
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_actions', type=int, default=60)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--action_mode', type=str, default='agibot_joint')
    parser.add_argument('--use_local_vlm', type=str, default=None)
    parser.add_argument('--use_proprio', type=int, default=0)
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--port', type=int, default=7722)
    args = parser.parse_args()
    
    server_model = build_xvla(
        device=args.device, 
        num_actions=args.num_actions, 
        pretrained=args.pretrained, 
        action_mode=args.action_mode, 
        use_local_vlm=args.use_local_vlm, 
        use_proprio=bool(args.use_proprio),
        version=args.version
    )
    
    server_model.run(port=args.port)
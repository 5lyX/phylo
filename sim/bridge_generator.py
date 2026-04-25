import os
import sys
import json
import argparse
import io
import contextlib
import traceback

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_type", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", default="DEFAULT")
    parser.add_argument("--question_type", default="numerical")
    args = parser.parse_args()

    suppressed_stdout = io.StringIO()
    suppressed_stderr = io.StringIO()
    try:
        # Suppress noisy library prints so stdout contains only JSON payload.
        with contextlib.redirect_stdout(suppressed_stdout), contextlib.redirect_stderr(suppressed_stderr):
            from sim.scene_generator import SceneGenerator
            generator = SceneGenerator(
                subtype=args.scene_type,
                seed=args.seed
            )
            scene_yaml = generator.generate_scene_yaml()

        # Output as JSON so the 3.12 process can parse stdout deterministically.
        sys.stdout.write(json.dumps(scene_yaml))
        sys.stdout.flush()

    except Exception as e:
        print(
            json.dumps(
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "suppressed_stdout": suppressed_stdout.getvalue(),
                    "suppressed_stderr": suppressed_stderr.getvalue(),
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)

if __name__ == "__main__":
    main()

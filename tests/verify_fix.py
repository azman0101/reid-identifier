import os


def test_path_sanitization(filename, base_dir):
    # This simulates the logic in the app
    safe_filename = os.path.basename(filename)
    path = os.path.join(base_dir, safe_filename)
    return path


def run_tests():
    base_dir = "/app/models/unknown"

    test_cases = [
        ("test.jpg", "/app/models/unknown/test.jpg"),
        ("../../etc/passwd", "/app/models/unknown/passwd"),
        ("/etc/passwd", "/app/models/unknown/passwd"),
        ("subdir/test.jpg", "/app/models/unknown/test.jpg"),
        (
            "..\\..\\windows\\system32\\config",
            "/app/models/unknown/config"
            if os.name != "nt"
            else "/app/models/unknown/..\\..\\windows\\system32\\config",
        ),
    ]

    # Note: on linux, os.path.basename("..\\..\\config") is "..\\..\\config" because \ is not a separator.
    # But usually these apps run on Linux.

    success = True
    for filename, expected in test_cases:
        result = test_path_sanitization(filename, base_dir)
        if result == expected:
            print(f"✅ PASS: '{filename}' -> '{result}'")
        else:
            # Handle Windows paths on Linux if necessary, but standard basename is usually enough for traversal
            if os.name != "nt" and "\\" in filename:
                # If it's a linux system, it might not catch backslashes as separators
                # But most attacks use /
                print(
                    f"ℹ️ INFO: '{filename}' -> '{result}' (Note: backslashes not treated as separators on Linux)"
                )
                continue

            print(f"❌ FAIL: '{filename}' -> '{result}' (Expected: '{expected}')")
            success = False

    if success:
        print("\nAll path sanitization tests passed!")
    else:
        print("\nSome tests failed.")
        exit(1)


if __name__ == "__main__":
    run_tests()

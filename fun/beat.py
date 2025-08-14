import pygame
import csv
import time


def main(audio_file, output_csv):
    pygame.init()
    pygame.mixer.init()

    # 创建一个小窗口，才能接收键盘事件
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption("按空格记录时间戳，按ESC退出")

    # 加载音频
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    timestamps = []

    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    start_time = time.time()
    running = True

    while running:
        screen.fill((30, 30, 30))
        text_surface = font.render(
            "播放中... 按空格记录，按ESC退出", True, (200, 200, 200)
        )
        screen.blit(text_surface, (20, 40))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current_time = time.time() - start_time
                    timestamps.append(current_time)
                    print(f"记录时间戳：{current_time:.3f} 秒")

                elif event.key == pygame.K_ESCAPE:
                    running = False

        clock.tick(60)

    pygame.mixer.music.stop()
    pygame.quit()

    # 保存时间戳
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_seconds"])
        for t in timestamps:
            writer.writerow([t])

    print(f"保存 {len(timestamps)} 条时间戳到 {output_csv}")


if __name__ == "__main__":
    audio_path = "/home/hy/wust_vision/fun/Gala - 你_H.ogg"  # 修改为你的音频路径
    output_csv = "user_timestamps.csv"
    main(audio_path, output_csv)

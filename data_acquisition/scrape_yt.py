import pytube
import os

if __name__ == '__main__':

    print(len(links))
    unique = set()
    for x in links:
        unique.add(x)
    print(len(unique))

    singer_dir = 'gdrive/My Drive/mp3_singers/'

    for idx, link in enumerate(unique):
        print("Processing song %i" % idx)
        full_link = yt_string_prefix + link
        yt = pytube.YouTube(full_link)
        file_base = 'track_{}'.format(idx)
        yt.streams.filter(only_audio=True).first().download(singer_dir, filename=file_base)

        out_f = 'gdrive/My\\ Drive/mp3_singers/{}'.format(file_base)

        os.system('ffmpeg -i {}.mp4 {}.mp3'.format(out_f, out_f))
        os.system('rm {}.mp4'.format(out_f))

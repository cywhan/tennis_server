from django.db import models
import datetime


class UploadedVideo(models.Model):
    title = models.CharField(max_length=200)
    play_date = models.DateField("play_date", default=datetime.date.today)
    processed_video_path = models.CharField(max_length=200, null=True)
    origin_video = models.FileField(upload_to='', null=True, verbose_name="")

    def __str__(self):
        return self.title

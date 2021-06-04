import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UploadedVideo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('play_date', models.DateField(default=datetime.date.today, verbose_name='play_date')),
                ('processed_video_path', models.CharField(max_length=200, null=True)),
                ('origin_video', models.FileField(null=True, upload_to='', verbose_name='')),
            ],
        ),
    ]

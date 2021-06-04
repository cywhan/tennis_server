from django.shortcuts import render, redirect
from .models import UploadedVideo
from .VideoUpload.track import track

form_id = ''

# Create your views here.
def upload(request):
    return render(request, 'video_server/upload.html')


def upload_create(request):
    global form_id
    form = UploadedVideo()
    form.title = request.POST['title']
    form.play_date = request.POST['date']
    form.processed_video_path = ''
    try:
        form.origin_video = request.FILES['video']
    except:
        print("can't find video")
    form.save()
    uploaded_video = UploadedVideo.objects.get(id=form.id)
    print(f'origin_path {uploaded_video.origin_video.path}')
    form.processed_video_path = track(uploaded_video.origin_video.path)
    print(f'form.processed_video_path {form.processed_video_path}')
    form.save()
    form_id = form.id
    return redirect('video_show')


def show_video(request):
    global form_id
    vid_info = UploadedVideo.objects.all()
    return render(request, 'video_server/video_show.html', {'vid_info': vid_info})

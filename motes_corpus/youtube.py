import pandas as pd
import math
import os
import json
from motes_corpus.modeling import MOTESCorpus
from youtube_transcript_api import YouTubeTranscriptApi

def search_youtube(youtube_api, q="children|kids|kid|teen|teens", pageToken=None, topicId=None, debug=False, order="relevance"):
    '''
    Take a query and an optional nextPageToken, and return the youtube.search().list() results parsed
    as a DataFrame.
    
    Returns: (nextPageToken, response)
    
    Quota cost: 100 units.
    '''
    request = youtube_api.search().list(
            part="snippet",
            q=q,
            pageToken=pageToken,
            safeSearch="strict",
            maxResults=50,
            videoCaption="closedCaption",
            type="video",
            regionCode="US",
            relevanceLanguage="en",
            topicId=topicId,
            order=order # date | rating | relevance | title | videoCount | viewCount
    )
    response = request.execute()
    if debug:
        return response
    df = search_results_to_df(response)
    try:
        return response['nextPageToken'], df
    except:
        print(response)
        return None, df

def scrape_uploads_for_playlist(youtube_api, playlist_id=None, pageToken=None, debug=False):
    '''
    Take a query and an optional nextPageToken, and return the youtube.search().list() results parsed
    as a DataFrame.
    
    Returns: (nextPageToken, response)
    
    Quota cost: 100 units.
    '''
    request = youtube_api.search().list(
            part="snippet",
            q=q,
            pageToken=pageToken,
            safeSearch="strict",
            maxResults=50,
            videoCaption="closedCaption",
            type="video",
            regionCode="US",
            relevanceLanguage="en",
            topicId=topicId,
            order=order # date | rating | relevance | title | videoCount | viewCount
    )
    response = request.execute()
    if debug:
        return response
    df = search_results_to_df(response)
    try:
        return response['nextPageToken'], df
    except:
        print(response)
        return None, df

def search_results_to_df(response, id_in='id'):
    '''id_in is the key which has the videoId within it'''
    parsed = []
    if 'items' not in response:
        return None
    for item in response['items']:
        try:
            parsed_item = parse_video_item(item, videoId=item[id_in]['videoId'])
            parsed.append(parsed_item)
        except:
            raise
            continue
    return pd.DataFrame(parsed)


def load_video_details(youtube_api, idlist, debug=False):
    '''
    Once ids for videos have been collected from search, you can ask for more details
    '''
    request = youtube_api.videos().list(
        part="status,topicDetails,contentDetails,id,snippet",
        id=",".join(idlist),
        maxResults=len(idlist)
    )
    response = request.execute()
    if debug:
        return response
    details = pd.DataFrame([parse_video_item(item, videoId=item['id']) for item in response['items']])
    return details

def augment_initial_search(youtube_api, df):
    details_collector = []
    for i in range(math.ceil(len(df)/50)):
        print(i, end=',')
        subset = df.iloc[i*50:i*50+50]
        details = load_video_details(youtube_api, subset.videoId.tolist())
        details_collector.append(details)
    all_details = pd.concat(details_collector)
    print()
    cols = ['videoId'] + [col for col in all_details.columns if col not in df.columns]
    return df.merge(all_details[cols], how='left', on='videoId')


def load_channel_info(youtube_api, channelIds):
    ''' Get details - most importantly the 'uploads' playlist - for a channel '''
    channel_collector = []
    
    for i in range(math.ceil(len(channelIds)/50)):
        print(i, end=',')
        subset = channelIds[i*50:i*50+50]
        
        request = youtube_api.channels().list(
            part='contentDetails,id,status,snippet',
            id=",".join(subset),
            maxResults=50
        )
        response = request.execute()
        channel_info = pd.DataFrame(parse_channel_item(item) for item in response['items'])
        channel_collector.append(channel_info)

    all_channel_info = pd.concat(channel_collector)
    return all_channel_info

def parse_channel_item(item, channelId=None):
    if channelId is None:
        channelId = item['id']
        
    details = dict(channelId=channelId)
        
    sections = {
        'snippet': ['title', 'description', 'customUrl', 'publishedAt'],
        'status': ['madeForKids']
    }
    
    for k, v in sections.items():
        if k in item:
            for k2 in v:
                if k2 in item[k]:
                    details[k2] = item[k][k2]
        
    for k in ['uploads', 'favorites', 'likes']:
        details[k] = item['contentDetails']['relatedPlaylists'][k]

    return details

def parse_video_item(item, videoId=None):
    if videoId == None:
        try:
            videoId = item['id']['videoId']
        except:
            videoId = item['contentDetails']['videoId']
    details = dict(videoId=videoId)

    sections = {
        'snippet': ['title', 'description', 'publishedAt', 'channelTitle', 'channelId', 'tags', 'categoryId'],
        'contentDetails': ['duration', 'caption'],
        'topicDetails': ['relevantTopicIds', 'topicCategories'],
        'status': ['madeForKids'],
        'statistics': ['viewCount', 'likeCount', 'dislikeCount', 'favoriteCount']
    }
    
    for k, v in sections.items():
        if k in item:
            for k2 in v:
                if k2 in item[k]:
                    details[k2] = item[k][k2]

    return details

def playlist_details(youtube_api, playlistId, debug=False):
    page_token = None
    detail_collector = []
    while True:
        request = youtube_api.playlistItems().list(
            part="id,snippet,contentDetails",
            playlistId=playlistId,
            pageToken = page_token,
            maxResults=50,
        )
        response = request.execute()
        if debug:
            return response
        df = search_results_to_df(response, id_in='contentDetails')
        try:
            detail_collector.append(df)
            page_token = response['nextPageToken']
        except:
            break
    all_details = pd.concat(detail_collector)
    return all_details


def fetch_transcript(videoId):
    transcripts = list(YouTubeTranscriptApi.list_transcripts(videoId))
    transcripts = [t for t in transcripts if 'english' in t.language.lower()]
    
    if len(transcripts) > 0:
        try:
            # try for first non-autogenerated, if it exists
            transcript = [t for t in transcripts if not t.is_generated][0]
        except:
            transcript = transcripts[0]
        text = transcript.fetch()
        return text
    else:
        return None
    
def fetch_and_save_transcript(videoId, save_dir, check_exists=True):
    fpath = os.path.join(save_dir, videoId+'.json')
    if check_exists and os.path.exists(fpath):
        return None
    
    text = fetch_transcript(videoId)
    with open(fpath, mode='w') as f:
        json.dump(text, f)
        
        
class YTCaption(object):
    
    pre_tokenized = False
    def __init__(self, path):
        try:
            with open(path) as f:
                self.json = json.load(f)
        except json.JSONDecodeError:
            self.json = []

        if not self.json:
            self.json = []
            
    def _basic_text(self, drop_newline=False):
        '''
        Simply concatenate all the text together.
        '''
        lines = [x['text'] for x in self.json if 'text' in x]
        txt = "\n".join(lines)
        if drop_newline:
            txt = txt.replace('\n', ' ').replace('  ', ' ')
        return txt
    
    def _clean_text(self, text):
        # Nothing Done Yet
        return text
    
    @property
    def text(self):
        txt = self._basic_text(drop_newline=True)
        txt = self._clean_text(txt)
        return txt


class YTCaptionCorpus(MOTESCorpus):
    
    DocClass = YTCaption
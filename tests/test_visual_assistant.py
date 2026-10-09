import asyncio, hashlib, io
from PIL import Image
from unittest.mock import patch, AsyncMock
from analytics import visual_assistant, assistant
import httpx

def test_visual_description(tmp_path):
    image=Image.new('RGB',(30,30),'gray'); buffer=io.BytesIO(); image.save(buffer,format='PNG'); content=buffer.getvalue(); sha=hashlib.sha256(content).hexdigest()
    record={'id':'test-image','source':'uploaded_image','image_sha256':sha,'quality':{'passed':True}}
    frame={'content':content,'sha256':sha}
    provider=AsyncMock(return_value=httpx.Response(200,json={'choices':[{'message':{'content':'A metallic rectangular component; exact identity is uncertain.'}}]}))
    with patch.object(assistant,'STORE',tmp_path),patch.object(assistant,'key',return_value='test-token'),patch.object(assistant,'provider_post',provider):
        result=asyncio.run(visual_assistant.describe(record,frame))
    assert result['read_only'] and result['image_sha256']==sha
    payload=provider.call_args.kwargs['json']; assert payload['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert list((tmp_path/'vision').glob('*.json'))
    assert record['quality']['passed']

def test_visual_description_rejects_wrong_image():
    import pytest
    with pytest.raises(assistant.AssistantUnavailable,match='integrity'):
        asyncio.run(visual_assistant.describe({'source':'uploaded_image','quality':{'passed':True},'image_sha256':'wrong'},{'sha256':'wrong','content':b'not-image'}))

def test_visual_description_rejects_bad_quality():
    import pytest
    with pytest.raises(assistant.AssistantUnavailable,match='Retake'):
        asyncio.run(visual_assistant.describe({'source':'uploaded_image','quality':{'passed':False}}, {'content':b'x'}))

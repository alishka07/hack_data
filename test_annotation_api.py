import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from annotation_api import atomic_json, register_annotation_routes


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.video={'video_id':'X1','source_sha256':'a'*64,'file_name':'X-1.mov',
                    'metadata':{'width':100,'height':200,'frames':3,'fps':10,'duration':.3},
                    'notes':[{'id':'note1','description':'test','timestamp_sec':.1}]}
        atomic_json(self.root/'catalog.json',{'videos':[self.video]})
        folder=self.root/'videos/X1';folder.mkdir(parents=True)
        (folder/'frames.jsonl').write_text('\n'.join(json.dumps({'frame_index':i,'timestamp_sec':i/10}) for i in range(3)))
        app=FastAPI();register_annotation_routes(app,self.root,review_root=self.root)
        self.client=TestClient(app)

    def tearDown(self):
        self.client.close();self.tmp.cleanup()

    def payload(self):
        return {'source_sha256':'a'*64,'revision':0,'annotation':{'kind':'object','status':'confirmed',
                'label':'cash','start_sec':.1,'end_sec':.1,'source_frame':1,'box':[10,10,30,20],
                'source_note_id':'note1','note':'visible banknote'}}

    def test_unknown_card_type_and_reflection_are_not_relabelled_as_identity_or_person(self):
        for revision, label in enumerate(('card', 'person_reflection')):
            payload=self.payload()
            payload['revision']=revision
            payload['annotation']['label']=label
            response=self.client.post('/api/annotations/X1/review',json=payload)
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['annotations'][-1]['label'],label)

    def write_video(self, source=None):
        atomic_json(self.root/'catalog.json', {'videos':[self.video], **({'source':source} if source else {})})

    def set_timeline(self, times, proof=True):
        path=self.root/'videos/X1/frames.jsonl'
        path.write_text('\n'.join(json.dumps({'frame_index':i,'timestamp_sec':t}) for i,t in enumerate(times)))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if proof:
            self.video['decode_proof']={'decoded_frames':len(times),'first_timestamp_sec':times[0],
                                       'last_timestamp_sec':times[-1],'source_timeline_end_sec':.3,
                                       'frame_ledger':'videos/X1/frames.jsonl','frame_ledger_sha256':digest}
            self.write_video()
        return digest

    def playback_map(self, times=(0,.1,.2)):
        ledger=self.set_timeline([0,.1,.2])
        preview=self.root/'previews/X1.mp4';preview.parent.mkdir(exist_ok=True)
        preview.write_bytes(b'test-preview-binding')
        value={'source_sha256':self.video['source_sha256'],'timestamps':list(times),
               'source_frame_ledger_sha256':ledger,'preview_sha256':hashlib.sha256(preview.read_bytes()).hexdigest(),
               'mapping':'same_frame_index_after_full_count_verification','verified':True}
        atomic_json(self.root/'previews/X1.timeline.json',value)
        return value

    def test_save_readback_preserves_source_and_requires_revision(self):
        before=(self.root/'catalog.json').read_bytes()
        first=self.client.post('/api/annotations/X1/review',json=self.payload())
        self.assertEqual(first.status_code,200,first.text)
        self.assertEqual(first.json()['revision'],1)
        self.assertEqual(first.json()['annotations'][0]['reviewer'],'user')
        self.assertFalse(first.json()['annotations'][0]['independent_test_eligible'])
        self.assertEqual(self.client.get('/api/annotations/X1').json()['review']['revision'],1)
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=self.payload()).status_code,409)
        self.assertEqual(before,(self.root/'catalog.json').read_bytes())

    def test_wrong_source_rejected(self):
        value=self.payload();value['source_sha256']='b'*64
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=value).status_code,409)

    def test_frame_time_mismatch_rejected(self):
        value=self.payload();value['annotation']['source_frame']=2
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=value).status_code,422)

    def test_box_outside_image_rejected(self):
        value=self.payload();value['annotation']['box']=[10,10,101,20]
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=value).status_code,422)

    def test_reversed_interval_rejected(self):
        value=self.payload();value['annotation'].update(kind='event',start_sec=.2,end_sec=.1)
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=value).status_code,422)

    def test_note_from_other_video_rejected(self):
        value=self.payload();value['annotation']['source_note_id']='foreign'
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=value).status_code,422)

    def test_source_frame_ledger_and_export(self):
        self.assertEqual(len(self.client.get('/api/annotations/X1/timeline').json()['frames']),3)
        exported=self.client.get('/api/annotations/export')
        self.assertEqual(exported.status_code,200)
        self.assertFalse(exported.json()['independent_test_eligible'])

    def test_asset_cannot_escape_catalog(self):
        response=self.client.get('/api/annotations/assets/%2e%2e%2fsecret.jpg')
        self.assertEqual(response.status_code,404)

    def test_exact_negative_preroll_object_frame_is_allowed(self):
        self.set_timeline([-1/30,0,.1])
        value=self.payload();value['annotation'].update(start_sec=-1/30,end_sec=-1/30,source_frame=0)
        response=self.client.post('/api/annotations/X1/review',json=value)
        self.assertEqual(response.status_code,200,response.text)
        saved=response.json()['annotations'][0]
        self.assertEqual(saved['source_frame'],0)
        self.assertEqual(saved['start_sec'],-1/30)

    def test_before_source_preroll_and_negative_event_rejected(self):
        self.set_timeline([-1/30,0,.1])
        value=self.payload();value['annotation'].update(start_sec=-.1,end_sec=-.1,source_frame=0)
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=value).status_code,422)
        value['annotation'].update(kind='event',start_sec=-1/30,end_sec=.1)
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=value).status_code,422)

    def test_playback_map_requires_valid_byte_and_ledger_bindings(self):
        mapping=self.playback_map()
        response=self.client.get('/api/annotations/X1/timeline')
        self.assertTrue(response.json()['playback_mapping_valid'])
        self.assertEqual(response.json()['frames'][2]['playback_timestamp_sec'],.2)
        for field, wrong in (('preview_sha256','b'*64),('source_frame_ledger_sha256','b'*64),
                             ('source_sha256','b'*64),('verified',False),('mapping','assumed_fps')):
            bad=dict(mapping);bad[field]=wrong
            atomic_json(self.root/'previews/X1.timeline.json',bad)
            result=self.client.get('/api/annotations/X1/timeline').json()
            self.assertFalse(result['playback_mapping_valid'],field)
            self.assertNotIn('playback_timestamp_sec',result['frames'][0])
        atomic_json(self.root/'previews/X1.timeline.json',mapping)
        (self.root/'previews/X1.mp4').write_bytes(b'replaced-preview')
        self.assertFalse(self.client.get('/api/annotations/X1/timeline').json()['playback_mapping_valid'])

    def test_playback_map_rejects_ambiguous_timestamps(self):
        mapping=self.playback_map()
        for times in ([0,0,.2],[0,.2,.1],[-.1,0,.1],[0,'0.1',.2],[False,.1,.2],[0,.1]):
            bad=dict(mapping);bad['timestamps']=times
            atomic_json(self.root/'previews/X1.timeline.json',bad)
            self.assertFalse(self.client.get('/api/annotations/X1/timeline').json()['playback_mapping_valid'])
        bad=dict(mapping);bad['timestamps']=[0,float('nan'),.2]
        (self.root/'previews/X1.timeline.json').write_text(json.dumps(bad))
        self.assertFalse(self.client.get('/api/annotations/X1/timeline').json()['playback_mapping_valid'])

    def test_missing_playback_map_is_explicit(self):
        value=self.client.get('/api/annotations/X1/timeline').json()
        self.assertFalse(value['playback_mapping_valid'])
        self.assertTrue(value['playback_mapping_reason'])

    def test_changed_frame_ledger_rejected(self):
        self.set_timeline([0,.1,.2])
        path=self.root/'videos/X1/frames.jsonl'
        path.write_text(path.read_text().replace('0.1','0.15'))
        self.assertEqual(self.client.get('/api/annotations/X1/timeline').status_code,409)
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=self.payload()).status_code,409)

    def test_noncontiguous_or_repeated_source_timeline_rejected(self):
        self.set_timeline([0,0,.2],proof=False)
        self.assertEqual(self.client.get('/api/annotations/X1/timeline').status_code,409)
        path=self.root/'videos/X1/frames.jsonl'
        path.write_text('\n'.join(json.dumps({'frame_index':i+1,'timestamp_sec':i/10}) for i in range(3)))
        self.assertEqual(self.client.get('/api/annotations/X1/timeline').status_code,409)

    def test_document_revision_required_and_note_provenance_saved(self):
        source={'document_id':'doc','document_url':'https://docs.google.com/document/d/doc','revision_id':'rev-1'}
        self.video['notes'][0]['provenance']=dict(source,row_index=2,tab_id='t.0')
        self.write_video(source)
        self.assertEqual(self.client.get('/api/annotations/X1').json()['source_document_revision_id'],'rev-1')
        value=self.payload()
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=value).status_code,409)
        value['source_document_revision_id']='rev-stale'
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=value).status_code,409)
        value['source_document_revision_id']='rev-1'
        response=self.client.post('/api/annotations/X1/review',json=value)
        self.assertEqual(response.status_code,200,response.text)
        annotation=response.json()['annotations'][0]
        self.assertEqual(annotation['source_note_provenance']['row_index'],2)
        self.assertEqual(annotation['source_document_revision_id'],'rev-1')

    def test_source_note_revision_mismatch_rejected(self):
        self.video['notes'][0]['provenance']={'revision_id':'stale'}
        self.write_video({'revision_id':'new'})
        self.assertEqual(self.client.get('/api/annotations/X1').status_code,409)

    def test_boolean_revision_rejected_on_save_and_remove(self):
        value=self.payload();value['revision']=False
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=value).status_code,409)
        saved=self.client.post('/api/annotations/X1/review',json=self.payload()).json()
        annotation_id=saved['annotations'][0]['id']
        remove={'source_sha256':'a'*64,'revision':True}
        self.assertEqual(self.client.post(f'/api/annotations/X1/review/{annotation_id}/remove',json=remove).status_code,409)

    def test_corrupted_review_is_not_silently_reset(self):
        path=self.root/'reviews/X1.json';path.parent.mkdir(exist_ok=True)
        path.write_text('{broken')
        before=path.read_bytes()
        self.assertEqual(self.client.post('/api/annotations/X1/review',json=self.payload()).status_code,409)
        self.assertEqual(path.read_bytes(),before)

    def test_manifest_source_mismatch_rejected(self):
        self.video['manifest_path']='videos/X1/manifest.json'
        self.write_video()
        atomic_json(self.root/self.video['manifest_path'],dict(self.video,source_sha256='b'*64))
        self.assertEqual(self.client.get('/api/annotations/X1').status_code,409)

    def test_changed_original_source_rejected(self):
        source=self.root/'original.mov';source.write_bytes(b'original-bytes')
        self.video.update(source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        self.write_video()
        self.assertEqual(self.client.get('/api/annotations/X1').status_code,200)
        source.write_bytes(b'changed-original-bytes')
        self.assertEqual(self.client.get('/api/annotations/X1').status_code,409)

    def test_export_keeps_source_url_and_duplicate_groups(self):
        source={'document_url':'https://docs.google.com/document/d/doc','revision_id':'rev-1'}
        group={'video_ids':['X1','X2'],'source_sha256':'a'*64,'label_conflict_status':'different_notes_require_review'}
        atomic_json(self.root/'catalog.json',{'videos':[self.video],'source':source,'duplicate_groups':[group]})
        self.assertEqual(self.client.get('/api/annotations/X1').json()['duplicate_group'],group)
        exported=self.client.get('/api/annotations/export').json()
        self.assertEqual(exported['source'],source)
        self.assertEqual(exported['duplicate_groups'],[group])


if __name__=='__main__':unittest.main()

"""Skin-only multipart ingress: bounded streaming, in-memory clinical uploads.

FastAPI parses multipart before its endpoint. This custom route replaces only
Skin POST requests with a capped Request, so neither MRI nor Voice parsing changes.
"""
from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import parse_options_header

from .config import get_settings


class BodyTooLarge(MultiPartException):
    pass


class MemoryMultipartParser(MultiPartParser):
    def __init__(self,*args,max_upload_bytes,**kwargs):
        super().__init__(*args,**kwargs)
        self.max_upload_bytes=max_upload_bytes
        # Body is capped BEFORE parser.write; this threshold cannot be reached.
        self.spool_max_size=max_upload_bytes+65536
        self._part_bytes=0

    def on_part_begin(self):
        self._part_bytes=0
        super().on_part_begin()

    def on_part_data(self,data,start,end):
        if self._current_part.file is not None:
            self._part_bytes+=end-start
            if self._part_bytes>self.max_upload_bytes:
                raise BodyTooLarge('The uploaded image exceeds the configured size limit.')
        super().on_part_data(data,start,end)


class SkinRequest(Request):
    def __init__(self,scope,receive,max_upload_bytes):
        super().__init__(scope,receive)
        self.max_upload_bytes=max_upload_bytes
        self.max_body_bytes=max_upload_bytes+65536

    async def stream(self):
        total=0
        async for chunk in super().stream():
            total+=len(chunk)
            if total>self.max_body_bytes:
                raise BodyTooLarge('The request exceeds the configured upload size limit.')
            yield chunk

    async def _get_form(self,*,max_files=1000,max_fields=1000,max_part_size=1024*1024):
        if self._form is not None:
            return self._form
        content_type,_=parse_options_header(self.headers.get('Content-Type'))
        try:
            if content_type==b'multipart/form-data':
                parser=MemoryMultipartParser(self.headers,self.stream(),max_files=1,max_fields=0,
                    max_part_size=65536,max_upload_bytes=self.max_upload_bytes)
                self._form=await parser.parse()
                return self._form
            return await super()._get_form(max_files=max_files,max_fields=max_fields,max_part_size=max_part_size)
        except BodyTooLarge as error:
            # Starlette closes every allocated in-memory spool on MultiPartException.
            raise HTTPException(413,'The uploaded image exceeds the configured size limit.') from error
        except MultiPartException as error:
            raise HTTPException(400,'Invalid multipart image upload.') from error


class SkinRoute(APIRoute):
    def get_route_handler(self):
        original=super().get_route_handler()
        async def bounded_handler(request):
            if request.method!='POST':
                return await original(request)
            try:
                settings=get_settings()
            except ValueError as error:
                raise HTTPException(503,'Skin configuration is invalid.') from error
            length=request.headers.get('content-length')
            if length is not None:
                try:
                    length=int(length)
                    if length<0:
                        raise ValueError()
                except ValueError as error:
                    raise HTTPException(400,'Invalid request size.') from error
                if length>settings.max_upload_bytes+65536:
                    raise HTTPException(413,'The request exceeds the configured upload size limit.')
            return await original(SkinRequest(request.scope,request.receive,settings.max_upload_bytes))
        return bounded_handler

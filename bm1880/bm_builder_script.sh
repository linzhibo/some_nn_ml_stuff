#!/usr/bin/env sh

/opt/bmtap2/bm1880-usb_1.0.3.3/bin/bm_builder.bin \
-t bm1880 \
-n alexnet \
-c bmnet_alexnet_int8.1x10.caffemodel \
--in_ctable=bmnet_alexnet_calibration_table.1x10.pb2 \
--out_ctable=bmnet_alexnet_out_calibration_table.1x10.pb2 \
--enable-weight-optimize=yes \
--enable-layer-group=yes \
--fc-left-shift=6 \
-s 1,3,227,227 \
-p ts_alexnet_frontend_opt.proto \
-o ts_alexnet.bmodel

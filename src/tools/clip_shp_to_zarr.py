import argparse
import os
import csv
import json
import numpy as np
import zarr
import fiona
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from rasterio.transform import array_bounds, Affine
from rasterio.warp import transform_geom

def build_bbox_polygon(store):
    h=store['observations'].shape[1]
    w=store['observations'].shape[2]
    t=store.attrs.get('transform')
    if isinstance(t,(list,tuple)):
        vals=list(t)
        if len(vals)>=6:
            aff=Affine(*vals[:6])
        else:
            aff=Affine.identity()
    else:
        aff=t
    left, bottom, right, top = array_bounds(h,w,aff)
    return Polygon([(left,bottom),(left,top),(right,top),(right,bottom)])

def read_label_mapping(csv_path):
    names=[]
    labels=[]
    with open(csv_path,'r',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        # find columns
        cols=reader.fieldnames
        # guess name column (first non 'label')
        name_col=cols[0] if cols and cols[0]!='label' else (cols[1] if len(cols)>1 else 'name')
        for row in reader:
            name=row.get(name_col)
            lab=row.get('label')
            if name is None or lab is None:
                continue
            try:
                lab=int(lab)
            except Exception:
                continue
            names.append(name)
            labels.append(lab)
    if not labels:
        return [],{}
    max_lab=max(labels)
    # convert to 0-based order by lab value
    label_names=[None]*(max_lab)
    for n,l in zip(names,labels):
        # assume labels start at 1
        idx=l-1
        if idx>=0:
            label_names[idx]=n
    # fill missing with string
    for i in range(len(label_names)):
        if label_names[i] is None:
            label_names[i]=f'class_{i}'
    return label_names,{n:(l-1) for n,l in zip(names,labels)}

def run(zarr_path, vector_path, csv_path, out_dir, min_count=0):
    os.makedirs(out_dir,exist_ok=True)
    store=zarr.open(zarr_path,mode='r')
    bbox_poly=build_bbox_polygon(store)
    zcrs=store.attrs.get('crs')
    label_names,name2idx=read_label_mapping(csv_path)
    counts={i:0 for i in range(len(label_names))}
    total=0
    feats=[]
    # prepare output schema
    with fiona.open(vector_path,'r') as src:
        vcrs=src.crs_wkt or src.crs
        schema=dict(src.schema)
        schema['properties']=dict(schema['properties'])
        schema['properties']['label']='int:10'
        for feat in src:
            geom=feat['geometry']
            # transform geom to zarr crs for containment check
            if vcrs and zcrs and vcrs!=zcrs:
                tgeom=transform_geom(vcrs,zcrs,geom)
            else:
                tgeom=geom
            poly=shape(tgeom)
            if poly.is_valid and bbox_poly.contains(poly):
                props=dict(feat['properties'])
                name=str(props.get('label_name', props.get('name', props.get('crop', 'unknown'))))
                if 'label' in props:
                    try:
                        idx=max(0,int(props['label']))
                    except Exception:
                        idx=name2idx.get(name,0)
                else:
                    idx=name2idx.get(name,0)
                feats.append({'geometry':feat['geometry'],'properties':props,'idx':idx})
                counts[idx]=counts.get(idx,0)+1
                total+=1
        # filter by min_count and reindex
        keep=[i for i,c in counts.items() if c>=min_count]
        old2new={old:i for i,old in enumerate(sorted(keep))}
        label_names_filtered=[label_names[old] for old in sorted(keep)]
        out_counts={str(i):0 for i in range(len(label_names_filtered))}
        out_path=os.path.join(out_dir,'polygons.shp')
        with fiona.open(out_path,'w',driver=src.driver,crs=src.crs,schema=schema) as dst:
            for f in feats:
                if f['idx'] in old2new:
                    new_idx=old2new[f['idx']]
                    props=dict(f['properties'])
                    props['label']=new_idx
                    dst.write({'geometry':f['geometry'],'properties':props})
                    out_counts[str(new_idx)]+=1
    # write polygons.json
    meta={
        'label_names':label_names_filtered,
        'num':{**out_counts,'sample':sum(out_counts.values())}
    }
    with open(os.path.join(out_dir,'polygons.json'),'w',encoding='utf-8') as f:
        json.dump(meta,f,ensure_ascii=False,indent=2)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zarr_path',type=str,required=True)
    p.add_argument('--vector_path',type=str,required=True)
    p.add_argument('--csv_path',type=str,required=True)
    p.add_argument('--out_dir',type=str,required=True)
    p.add_argument('--min_count',type=int,default=0)
    a=p.parse_args()
    run(a.zarr_path,a.vector_path,a.csv_path,a.out_dir,a.min_count)

if __name__=='__main__':
    main()

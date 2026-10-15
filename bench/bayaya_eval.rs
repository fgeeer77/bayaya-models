//! Runs translator-rs's still-image PP-OCR path (detect, dewarp, recognise, group into paragraphs)
//! over a list of pictures and prints one JSON line per picture. The same steps as
//! `ocr_runtime::translate_image_rgba_ppocr_in_snapshot` with a fixed script and no ink model,
//! minus the translation itself.
//!
//! Usage: bayaya_eval <det.mnn> <latin.mnn> <latin_keys.txt> <eslav.mnn> <eslav_keys.txt>
//!                    <max_image_size> <min_confidence 0-100> <manifest.tsv> <out.jsonl>
//! manifest lines: name \t image path \t latin|eslav

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use translator::PpocrScript;
use translator::coords::Quadrant;
use translator::ocr::{OrientedRect, TextLine, group_lines_into_paragraphs};
use translator::ocr_runtime::{build_still_image, detect_boxes_from_still};
use translator::ppocr::{PpocrEngine, PpocrProfile, PpocrRecognizerSpec};

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() == 10, "see usage in the source");
    let engine = PpocrEngine::load(
        &PathBuf::from(&a[1]),
        None,
        None,
        vec![
            PpocrRecognizerSpec { script: PpocrScript::Latin, model_path: a[2].clone().into(), keys_path: a[3].clone().into() },
            PpocrRecognizerSpec { script: PpocrScript::Eslav, model_path: a[4].clone().into(), keys_path: a[5].clone().into() },
        ],
        4,
        None,
    )
    .expect("load engine");
    let max_image: u32 = a[6].parse().unwrap();
    let min_conf: f32 = a[7].parse::<f32>().unwrap() / 100.0;
    let manifest = std::fs::read_to_string(&a[8]).unwrap();
    let mut out = std::fs::File::create(&a[9]).unwrap();
    for row in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let cols: Vec<&str> = row.split('\t').collect();
        let (name, path, script) = (cols[0], cols[1], if cols[2] == "eslav" { PpocrScript::Eslav } else { PpocrScript::Latin });
        let mut img = image::open(path).expect("open image").to_rgba8();
        // The app hands Android ARGB pixels over as B,G,R,A bytes: do the same.
        for p in img.pixels_mut() {
            p.0.swap(0, 2);
        }
        let (w, h) = img.dimensions();
        let t0 = Instant::now();
        let still = build_still_image(img.as_raw(), w, h, max_image).expect("still");
        let boxes = detect_boxes_from_still(&engine, &still, w, h).expect("detect");
        let scripts = vec![script; boxes.len()];
        let strips = engine.dewarp_strips(&still.rgb, &boxes, Some(Quadrant::R0));
        let lines = engine
            .recognize_from_strips(&strips, &boxes, &scripts, PpocrProfile::Still, w, h, 0.0)
            .expect("recognize");
        let text_lines: Vec<TextLine> = boxes
            .iter()
            .zip(lines)
            .filter(|(_, l)| !l.text.trim().is_empty() && l.confidence >= min_conf)
            .map(|(b, l)| TextLine {
                text: l.text,
                bounding_box: l.rect,
                oriented_box: l.oriented_box,
                tight_box: OrientedRect { angle_radians: l.oriented_box.angle_radians, ..b.tight_box },
                word_rects: vec![l.rect],
                style_ranges: vec![],
                ink_color: None,
            })
            .collect();
        let blocks = group_lines_into_paragraphs(text_lines, Default::default());
        let ms = t0.elapsed().as_millis();
        let mut json = format!("{{\"name\":\"{}\",\"ms\":{},\"boxes\":{},\"blocks\":[", esc(name), ms, boxes.len());
        for (i, b) in blocks.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!("{{\"text\":\"{}\",\"lines\":[", esc(&b.translation_text())));
            for (j, l) in b.lines.iter().enumerate() {
                if j > 0 {
                    json.push(',');
                }
                let r = &l.bounding_box;
                json.push_str(&format!("{{\"text\":\"{}\",\"box\":[{},{},{},{}]}}", esc(&l.text), r.left, r.top, r.right, r.bottom));
            }
            json.push_str("]}");
        }
        json.push_str("]}");
        writeln!(out, "{json}").unwrap();
        eprintln!("{name}: {} boxes, {} blocks, {ms} ms", boxes.len(), blocks.len());
    }
}

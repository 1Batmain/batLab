#!/usr/bin/env python3
"""Tests des convertisseurs de datasets — aucun réseau, aucun fichier du dépôt.

    python3 tools/test_dataset_tools.py

Toutes les images sont fabriquées ici, à la main : la suite doit tourner sur une
machine qui n'a ni CIFAR, ni ImageNet, ni internet. Ce qu'elle vérifie n'est pas
« le script s'exécute » mais les propriétés dont dépend un entraînement — les
géométries, l'orientation EXIF, le déterminisme, et le fait qu'un aller-retour
rende les pixels de départ.
"""

import os
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import batraw
import images_to_raw
from PIL import Image


def solid(path, size, colour, fmt=None, exif=None):
    """Écrit une image unie. `colour` est un triplet RGB ou un entier (gris)."""
    mode = "L" if isinstance(colour, int) else "RGB"
    image = Image.new(mode, size, colour)
    kwargs = {}
    if exif is not None:
        kwargs["exif"] = exif
    image.save(path, format=fmt, **kwargs)
    return path


class BatrawFormat(unittest.TestCase):
    """Le format lui-même : trois encodages, les mêmes images."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="batlab_batraw_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_round_trip_returns_the_bytes_it_was_given(self):
        """Écrire puis relire rend les octets de départ, sur les trois versions.

        Pour BATRAW1 et 2 le trajet passe par des f32 : c'est donc aussi la
        preuve que la quantisation inverse est exacte, et pas seulement proche.
        """
        source = bytes(range(256)) * 3  # 768 octets = 16×16×3
        for version in (1, 2, 3):
            path = os.path.join(self.dir, f"v{version}.batraw")
            batraw.write(path, source, 1, 16, 16, 3, version=version)
            back, count, w, h, c, got = batraw.read(path)
            self.assertEqual((count, w, h, c, got), (1, 16, 16, 3, version))
            self.assertEqual(bytes(back), source, f"aller-retour cassé en BATRAW{version}")

    def test_the_8_bit_payload_is_a_quarter_of_the_f32_one(self):
        source = bytes(64)
        sizes = {}
        for version in (2, 3):
            path = os.path.join(self.dir, f"size{version}.batraw")
            sizes[version] = batraw.write(path, source, 1, 8, 8, 1, version=version)
        payload = lambda v: sizes[v] - batraw.HEADER_BYTES  # noqa: E731
        self.assertEqual(payload(2), payload(3) * 4)

    def test_the_header_says_what_the_engine_reads(self):
        """L'en-tête est lu par le Rust ; sa disposition est un contrat."""
        path = os.path.join(self.dir, "header.batraw")
        batraw.write(path, bytes(2 * 12), 2, 2, 2, 3, version=3)
        with open(path, "rb") as fh:
            raw = fh.read(batraw.HEADER_BYTES)
        self.assertEqual(raw[:8], b"BATRAW3\0")
        self.assertEqual(struct.unpack("<IIII", raw[8:24]), (2, 2, 2, 3))

    def test_a_payload_that_does_not_match_the_header_is_refused(self):
        path = os.path.join(self.dir, "ragged.batraw")
        with self.assertRaises(ValueError):
            batraw.write(path, bytes(10), 2, 2, 2, 3, version=3)


class ImagesToRaw(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="batlab_images_")
        self.src = os.path.join(self.dir, "src")
        os.makedirs(self.src)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def out(self, name="out.batraw"):
        return os.path.join(self.dir, name)

    def convert(self, **kwargs):
        options = dict(
            root=self.src,
            out_path=self.out(),
            size=8,
            mode="rgb",
            crop="center",
            min_size=0,
            dedup=False,
            version=3,
            verbose=False,
        )
        options.update(kwargs)
        return images_to_raw.convert(**options)

    # -- géométries ------------------------------------------------------

    def test_every_sample_has_the_geometry_that_was_asked_for(self):
        solid(os.path.join(self.src, "a.png"), (100, 40), (200, 10, 10))
        solid(os.path.join(self.src, "b.png"), (7, 250), (10, 200, 10))
        for mode, channels in (("rgb", 3), ("grey", 1)):
            report = self.convert(mode=mode, size=8, out_path=self.out(f"{mode}.batraw"))
            payload, count, w, h, c, _ = batraw.read(self.out(f"{mode}.batraw"))
            self.assertEqual(report["written"], 2)
            self.assertEqual((count, w, h, c), (2, 8, 8, channels))
            self.assertEqual(len(payload), 2 * 8 * 8 * channels)

    def test_a_centred_crop_keeps_the_middle_and_fit_keeps_everything(self):
        """Les deux recadrages doivent être distinguables sur une image conçue pour.

        Une bande rouge verticale au centre d'un fond bleu très large : `center`
        ne doit voir que du rouge, `fit` doit garder du bleu.
        """
        image = Image.new("RGB", (300, 60), (0, 0, 255))
        image.paste(Image.new("RGB", (60, 60), (255, 0, 0)), (120, 0))
        image.save(os.path.join(self.src, "band.png"))

        pixels = {}
        for crop in ("center", "fit"):
            self.convert(crop=crop, size=8, out_path=self.out(f"{crop}.batraw"))
            payload, _, _, _, _, _ = batraw.read(self.out(f"{crop}.batraw"))
            pixels[crop] = bytes(payload)

        centre_is_red = all(pixels["center"][i] > pixels["center"][i + 2] for i in range(0, 192, 3))
        self.assertTrue(centre_is_red, "--crop center a gardé autre chose que le centre")
        self.assertIn(
            True,
            [pixels["fit"][i + 2] > pixels["fit"][i] for i in range(0, 192, 3)],
            "--crop fit a perdu les bords de l'image",
        )

    def test_fit_pads_with_the_neutral_byte(self):
        """Les marges de `fit` doivent être le 0 du modèle, pas du noir.

        Du noir vaut -1 en [-1,1] : c'est un bord dur que le modèle apprendrait
        comme une caractéristique du dataset.
        """
        solid(os.path.join(self.src, "wide.png"), (64, 8), (255, 255, 255))
        self.convert(crop="fit", size=8)
        payload, _, _, _, _, _ = batraw.read(self.out())
        # Ligne du haut : entièrement de la marge.
        self.assertEqual(set(payload[: 8 * 3]), {images_to_raw.NEUTRAL})

    # -- EXIF ------------------------------------------------------------

    def test_the_exif_orientation_tag_is_applied(self):
        """Une photo de téléphone en portrait est stockée en paysage + un tag.

        Sans `exif_transpose`, tout un dataset de photos s'entraîne couché — et
        rien ne le signale. L'image ici est asymétrique haut/bas ; après rotation
        déclarée par le tag, le haut et le bas doivent avoir échangé.
        """
        image = Image.new("RGB", (32, 32), (0, 0, 0))
        image.paste(Image.new("RGB", (32, 16), (255, 255, 255)), (0, 0))  # blanc en haut

        exif = Image.Exif()
        exif[274] = 3  # Orientation = rotation de 180°
        image.save(os.path.join(self.src, "rotated.jpg"), exif=exif)

        self.convert(size=8, out_path=self.out("exif.batraw"))
        payload, _, _, _, _, _ = batraw.read(self.out("exif.batraw"))
        top_left = payload[0]
        bottom_left = payload[(7 * 8) * 3]
        self.assertLess(top_left, 64, "le tag EXIF n'a pas été appliqué (haut resté blanc)")
        self.assertGreater(bottom_left, 192, "le bas aurait dû devenir blanc")

    # -- robustesse ------------------------------------------------------

    def test_a_corrupt_file_is_skipped_and_counted(self):
        solid(os.path.join(self.src, "good.png"), (32, 32), (1, 2, 3))
        with open(os.path.join(self.src, "truncated.jpg"), "wb") as fh:
            fh.write(b"\xff\xd8\xff\xe0" + b"garbage" * 10)
        with open(os.path.join(self.src, "notanimage.png"), "wb") as fh:
            fh.write(b"this is plainly not a png")

        report = self.convert()
        self.assertEqual(report["written"], 1)
        self.assertEqual(report["unreadable"], 2)
        self.assertEqual(report["found"], 3)

    def test_min_size_drops_the_images_that_would_be_mush(self):
        solid(os.path.join(self.src, "big.png"), (64, 64), (1, 2, 3))
        solid(os.path.join(self.src, "tiny.png"), (20, 15), (4, 5, 6))

        # size=32 : le 20×15 est bien la « bouillie » que --min-size vise.
        loose = self.convert(min_size=0, size=32, out_path=self.out("loose.batraw"))
        self.assertEqual(loose["written"], 2)
        self.assertEqual(loose["upscaled"], 1, "l'agrandissement doit être compté et dit")

        strict = self.convert(min_size=32, size=32, out_path=self.out("strict.batraw"))
        self.assertEqual(strict["written"], 1)
        self.assertEqual(strict["too_small"], 1)

    def test_dedup_compares_the_sample_not_the_file(self):
        """Deux encodages du même cliché sont un seul échantillon.

        C'est le point : dédupliquer sur les octets du fichier laisserait passer
        la paire PNG/JPEG que tout scrape ramène, et le modèle verrait l'image
        deux fois.
        """
        image = Image.new("RGB", (64, 64), (30, 90, 150))
        image.save(os.path.join(self.src, "a.png"))
        image.save(os.path.join(self.src, "b.png"))
        image.resize((128, 128), Image.NEAREST).save(os.path.join(self.src, "c.png"))
        solid(os.path.join(self.src, "d.png"), (64, 64), (200, 10, 10))

        without = self.convert(dedup=False, out_path=self.out("dup.batraw"))
        self.assertEqual(without["written"], 4)

        with_dedup = self.convert(dedup=True, out_path=self.out("dedup.batraw"))
        self.assertEqual(with_dedup["written"], 2)
        self.assertEqual(with_dedup["duplicates"], 2)

    def test_an_empty_or_unusable_folder_is_an_error(self):
        with self.assertRaises(ValueError):
            self.convert()

    # -- déterminisme et aller-retour -----------------------------------

    def test_two_runs_over_the_same_folder_give_the_same_file(self):
        """L'ordre vient du chemin, pas de l'ordre du système de fichiers.

        Créés dans un ordre, nommés dans un autre : si le tri n'était pas fait,
        les deux exécutions différeraient sur cette machine ou sur une autre.
        """
        os.makedirs(os.path.join(self.src, "z"))
        os.makedirs(os.path.join(self.src, "a"))
        solid(os.path.join(self.src, "z", "2.png"), (32, 32), (10, 20, 30))
        solid(os.path.join(self.src, "a", "1.png"), (32, 32), (40, 50, 60))
        solid(os.path.join(self.src, "m.png"), (32, 32), (70, 80, 90))

        first = self.out("first.batraw")
        second = self.out("second.batraw")
        self.convert(out_path=first)
        self.convert(out_path=second)
        with open(first, "rb") as a, open(second, "rb") as b:
            self.assertEqual(a.read(), b.read())

        # …et l'ordre est bien celui des chemins triés : a/1, m, z/2.
        payload, _, _, _, _, _ = batraw.read(first)
        self.assertEqual(payload[0:3], bytes((40, 50, 60)))
        self.assertEqual(payload[192:195], bytes((70, 80, 90)))
        self.assertEqual(payload[384:387], bytes((10, 20, 30)))

    def test_a_flat_colour_survives_the_whole_trip(self):
        """L'aller-retour complet : image → .batraw → pixels.

        Une couleur unie traverse le recadrage et le redimensionnement sans
        changer, donc tout écart est un défaut du pipeline et pas du filtre de
        rééchantillonnage. Vérifié sur les trois formats : c'est aussi la preuve
        que le choix de `--format` ne change pas les images.
        """
        solid(os.path.join(self.src, "flat.png"), (40, 40), (17, 200, 99))
        for version in (1, 2, 3):
            path = self.out(f"trip{version}.batraw")
            self.convert(version=version, out_path=path)
            payload, count, w, h, c, _ = batraw.read(path)
            self.assertEqual((count, w, h, c), (1, 8, 8, 3))
            self.assertEqual(set(payload[0::3]), {17})
            self.assertEqual(set(payload[1::3]), {200})
            self.assertEqual(set(payload[2::3]), {99})

    def test_grey_uses_the_same_luminance_as_the_cifar_converter(self):
        """Les deux convertisseurs doivent produire le même gris.

        Sinon un fine-tune de CIFAR vers des images à soi change la définition
        de « gris » en cours de route.
        """
        solid(os.path.join(self.src, "colour.png"), (32, 32), (200, 100, 50))
        self.convert(mode="grey", out_path=self.out("grey.batraw"))
        payload, _, _, _, _, _ = batraw.read(self.out("grey.batraw"))
        expected = round(0.299 * 200 + 0.587 * 100 + 0.114 * 50)
        self.assertLessEqual(
            abs(payload[0] - expected), 1, f"gris {payload[0]}, BT.601 donne {expected}"
        )

    # -- planche ---------------------------------------------------------

    def test_the_contact_sheet_shows_the_whole_dataset_not_its_head(self):
        """La planche échantillonne tout le dataset.

        Un scrape trié par chemin a ses 64 premiers fichiers dans le même
        sous-dossier ; une planche qui ne montrerait qu'eux mentirait par
        omission. Ici la dernière image est d'une couleur unique — elle doit
        apparaître.
        """
        for i in range(200):
            solid(os.path.join(self.src, f"{i:03d}.png"), (32, 32), (i, i, i))
        solid(os.path.join(self.src, "zzz.png"), (32, 32), (255, 0, 0))

        self.convert(size=8)
        png = os.path.join(self.dir, "sheet", "plate.png")
        shown = images_to_raw.contact_sheet(self.out(), png, columns=4, cell=16, rows=4)

        self.assertEqual(shown, 16)
        self.assertTrue(os.path.exists(png))
        sheet = Image.open(png)
        self.assertEqual(sheet.size, (4 * 16, 4 * 16))
        colours = [c for _, c in sheet.convert("RGB").getcolors(4096)]
        self.assertIn((255, 0, 0), colours, "la dernière image du dataset n'est pas sur la planche")
        self.assertIn((0, 0, 0), colours, "la première image du dataset n'est pas sur la planche")

    def test_the_contact_sheet_works_in_greyscale_too(self):
        solid(os.path.join(self.src, "a.png"), (32, 32), (10, 10, 10))
        self.convert(mode="grey", size=8)
        png = os.path.join(self.dir, "grey_sheet.png")
        images_to_raw.contact_sheet(self.out(), png, columns=2, cell=8, rows=2)
        self.assertEqual(Image.open(png).mode, "L")


if __name__ == "__main__":
    unittest.main(verbosity=2)

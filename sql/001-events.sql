create schema if not exists "cda_fact_extraction_202603";

-- 134,951,266 rows
create table if not exists cda_fact_extraction_202603.events (
  id bigserial primary key,
  data jsonb,
  hidden boolean default false,
  epi_id text,
  epi_type text,
  effective_time timestamp,
  texts_id int,
  schema text,
  "table" text,
  field text,
  table_id text,
  header text,
  header_offset int,
  event_offset int,
  event_header_date text,
  doctor_code text,
  specialty text,
  specialty_code text
);

insert into cda_fact_extraction_202603.events (
  data,
  hidden,
  epi_id,
  epi_type,
  effective_time,
  texts_id,
  schema,
  "table",
  field,
  table_id,
  header,
  header_offset,
  event_offset,
  event_header_date,
  doctor_code,
  specialty,
  specialty_code
)
select
  case
    when n = 13 then null
    else jsonb_build_object(
      'text',
      case
        when n in (26, 27) then ''
        when (n - 1) % 25 < 13 then
          (array[
            'Vererõhk 140/90, pulss 78.',
            'Diagnoositud essentsiaalne hüpertensioon.',
            'Patsient suunatud kardioloogi vastuvõtule.',
            'Kaebused rindkere valule füüsilisel koormusel.',
            'EKG: siinusrütm, sagedus 72.',
            'Teostatud vereanalüüsid, CRP normis.',
            'Soovitatud elustiili muutused ja soolavaba dieet.',
            'Kontrollvisiit kolme kuu pärast.',
            'Patsiendil esineb õhupuudus trepist käimisel.',
            'Kopsudes kuulda nõrgenenud hingamiskahinat.',
            'Ravimite taluvus hea, kõrvaltoimeid ei esine.',
            'Teostatud ultraheliuuring kõhukoopast.',
            'Patsient suunatud kirurgi konsultatsioonile.',
            'Diabeediravi toimib, HbA1c paranenud.',
            'Kolesterool normis, kuur lõpetatud plaanipäraselt.',
            'Patsient jätkab statiinravi, taluvus hea.'
          ])[1 + (n - 1) % 16]
        else
          (array[
            'Patsient katkestas losartaani võtmise, köha tõttu.',
            'Ravi lõpetas arst, vererõhk normaliseerus.',
            'Arst asendas amlodipiini teise ravimiga, tursete tõttu.',
            'Patsient vahetas statiini, lihasvalude tõttu.',
            'Patsient loobus varfariinist, veritsusohu tõttu.',
            'Annust vähendas arst, pearingluse tõttu.',
            'Ravi katkestamine arsti otsusel, maksaproovide tõusu tõttu.',
            'Metformiini lõpetamine seedehäirete tõttu.',
            'Ravimite taluvus hea, kõrvaltoimeid ei esine.',
            'Diabeediravi toimib, HbA1c paranenud.',
            'Vererõhk 140/90, pulss 78.',
            'EKG: siinusrütm, sagedus 72.'
          ])[1 + (n - 1) % 12]
      end,
      'meta', jsonb_build_object('source', 'mock', 'row', n),
      'layers', '[]'::jsonb,
      'relation_layers', '[]'::jsonb
    )
  end,
  false,
  n::text,
  case
    when n in (246, 247) then null
    when (n - 1) % 10 < 6 then 'ravim'
    when (n - 1) % 10 = 6 then 'saatekiri'
    when (n - 1) % 10 = 7 then 'kaebus'
    when (n - 1) % 10 = 8 then 'protseduur'
    else 'diagnoos'
  end,
  timestamp '2024-01-01 08:00:00' + (n - 1) * interval '2 hours',
  1000 + n,
  'cda',
  case when (n - 1) % 25 < 13 then 'document' else 'anamnesis' end,
  case
    when (n - 1) % 25 < 13 then 'narrative'
    when n % 2 = 0 then 'anamnesis'
    else 'anamnsum'
  end,
  (5000 + n)::text,
  case
    when (n - 1) % 25 < 13 then
      (array['Epikriis', 'Saatekiri', 'Läbivaatus', 'Vastuvõtt'])[1 + (n - 1) % 4]
    else 'Anamnees'
  end,
  (n * 7) % 120,
  (n * 13) % 250,
  (timestamp '2024-01-01 08:00:00' + (n - 1) * interval '2 hours')::date::text,
  'D00' || (1 + (n - 1) % 9),
  (array['sisehaigused', 'kardioloogia', 'neuroloogia', 'perearstiabi', 'onkoloogia'])[1 + (n - 1) % 5],
  (array['SISE', 'KARD', 'NEUR', 'PERE', 'ONK'])[1 + (n - 1) % 5]
from generate_series(1, 5000) as n;

create index if not exists "ix_events_epi_id"
on cda_fact_extraction_202603.events
using btree
(
  epi_id
);

create index if not exists "ix_events_epi_type"
on cda_fact_extraction_202603.events
using btree
(
  epi_type
);

create index if not exists "ix_events_layers_gin"
on cda_fact_extraction_202603.events
using gin
(
  (to_tsvector('simple', data ->> 'layers'))
);

create index
if not exists "ix_events_relation_layers_gin"
on cda_fact_extraction_202603.events
using gin
(
  (to_tsvector('simple', data ->> 'relation_layers'))
);

create schema if not exists "user_sven";

-- 16,720,717 rows
create table if not exists user_sven.dg_observed_cda_documents (
  patient_id bigint,
  index_drug text,
  codrug text,
  episode bigint,
  index_event date,
  epi_id text,
  epi_type text,
  start_date date,
  document_offset int,
  time_source text,
  event_type text
);

with
numbered
as (
  select
    events.epi_id,
    events.epi_type,
    events.effective_time,
    ROW_NUMBER() over (order by events.id) as n
  from
    cda_fact_extraction_202603.events as events
  where
    events."table" = 'anamnesis'
)
insert into user_sven.dg_observed_cda_documents (
  patient_id,
  index_drug,
  codrug,
  episode,
  index_event,
  epi_id,
  epi_type,
  start_date,
  document_offset,
  time_source,
  event_type
)
select
  -- prime modulus keeps patients decoupled from the phrase and event type cycles
  7000 + (n - 1) % 151 as patient_id,
  (array['A', 'B', 'C', 'D', 'E'])[1 + (n - 1) % 5]
  as index_drug,
  (array['F', 'G', 'H', 'I', 'J'])[1 + (n - 1) % 5]
  as codrug,
  1 + (n - 1) % 3 as episode,
  cast(effective_time as date) as index_event,
  epi_id,
  epi_type,
  cast(effective_time as date) - 30 as start_date,
  0 as document_offset,
  'mock' as time_source,
  (array['discontinuation', 'switch', 'usage_gap'])[1 + (n - 1) % 3]
  as event_type
from
  numbered;

create index
if not exists "ix_dg_observed_cda_documents_document_offset"
on user_sven.dg_observed_cda_documents
using btree
(
  document_offset
);

create index
if not exists "ix_dg_observed_cda_documents_epi_id"
on user_sven.dg_observed_cda_documents
using btree
(
  epi_id
);

create index
if not exists "ix_dg_observed_cda_documents_index_event_episode"
on user_sven.dg_observed_cda_documents
using btree
(
  index_event,
  episode
);

create index
if not exists "ix_dg_observed_cda_documents_index_drug_codrug"
on user_sven.dg_observed_cda_documents
using btree
(
  index_drug,
  codrug
);

create index
if not exists "ix_dg_observed_cda_documents_patient_id"
on user_sven.dg_observed_cda_documents
using btree
(
  patient_id
);

create index
if not exists "ix_dg_observed_cda_documents_start_date"
on user_sven.dg_observed_cda_documents
using btree
(
  start_date
);

create index
if not exists "ix_dg_observed_cda_documents_event_type"
on user_sven.dg_observed_cda_documents
using btree
(
  event_type
);

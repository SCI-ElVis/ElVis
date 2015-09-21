// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: RenderModule.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "RenderModule.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace ElVis {
namespace Serialization {

namespace {

const ::google::protobuf::Descriptor* RenderModule_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  RenderModule_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_RenderModule_2eproto() {
  protobuf_AddDesc_RenderModule_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "RenderModule.proto");
  GOOGLE_CHECK(file != NULL);
  RenderModule_descriptor_ = file->message_type(0);
  static const int RenderModule_offsets_[2] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(RenderModule, enabled_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(RenderModule, concrete_module_),
  };
  RenderModule_reflection_ =
    ::google::protobuf::internal::GeneratedMessageReflection::NewGeneratedMessageReflection(
      RenderModule_descriptor_,
      RenderModule::default_instance_,
      RenderModule_offsets_,
      -1,
      -1,
      -1,
      sizeof(RenderModule),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(RenderModule, _internal_metadata_),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(RenderModule, _is_default_instance_));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_RenderModule_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
      RenderModule_descriptor_, &RenderModule::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_RenderModule_2eproto() {
  delete RenderModule::default_instance_;
  delete RenderModule_reflection_;
}

void protobuf_AddDesc_RenderModule_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::protobuf_AddDesc_google_2fprotobuf_2fany_2eproto();
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\022RenderModule.proto\022\023ElVis.Serializatio"
    "n\032\031google/protobuf/any.proto\"N\n\014RenderMo"
    "dule\022\017\n\007enabled\030\001 \001(\010\022-\n\017concrete_module"
    "\030\002 \001(\0132\024.google.protobuf.Anyb\006proto3", 156);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "RenderModule.proto", &protobuf_RegisterTypes);
  RenderModule::default_instance_ = new RenderModule();
  RenderModule::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_RenderModule_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_RenderModule_2eproto {
  StaticDescriptorInitializer_RenderModule_2eproto() {
    protobuf_AddDesc_RenderModule_2eproto();
  }
} static_descriptor_initializer_RenderModule_2eproto_;

namespace {

static void MergeFromFail(int line) GOOGLE_ATTRIBUTE_COLD;
static void MergeFromFail(int line) {
  GOOGLE_CHECK(false) << __FILE__ << ":" << line;
}

}  // namespace


// ===================================================================

#ifndef _MSC_VER
const int RenderModule::kEnabledFieldNumber;
const int RenderModule::kConcreteModuleFieldNumber;
#endif  // !_MSC_VER

RenderModule::RenderModule()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:ElVis.Serialization.RenderModule)
}

void RenderModule::InitAsDefaultInstance() {
  _is_default_instance_ = true;
  concrete_module_ = const_cast< ::google::protobuf::Any*>(&::google::protobuf::Any::default_instance());
}

RenderModule::RenderModule(const RenderModule& from)
  : ::google::protobuf::Message(),
    _internal_metadata_(NULL) {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:ElVis.Serialization.RenderModule)
}

void RenderModule::SharedCtor() {
    _is_default_instance_ = false;
  _cached_size_ = 0;
  enabled_ = false;
  concrete_module_ = NULL;
}

RenderModule::~RenderModule() {
  // @@protoc_insertion_point(destructor:ElVis.Serialization.RenderModule)
  SharedDtor();
}

void RenderModule::SharedDtor() {
  if (this != default_instance_) {
    delete concrete_module_;
  }
}

void RenderModule::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* RenderModule::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return RenderModule_descriptor_;
}

const RenderModule& RenderModule::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_RenderModule_2eproto();
  return *default_instance_;
}

RenderModule* RenderModule::default_instance_ = NULL;

RenderModule* RenderModule::New(::google::protobuf::Arena* arena) const {
  RenderModule* n = new RenderModule;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void RenderModule::Clear() {
  enabled_ = false;
  if (GetArenaNoVirtual() == NULL && concrete_module_ != NULL) delete concrete_module_;
  concrete_module_ = NULL;
}

bool RenderModule::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:ElVis.Serialization.RenderModule)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional bool enabled = 1;
      case 1: {
        if (tag == 8) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   bool, ::google::protobuf::internal::WireFormatLite::TYPE_BOOL>(
                 input, &enabled_)));

        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(18)) goto parse_concrete_module;
        break;
      }

      // optional .google.protobuf.Any concrete_module = 2;
      case 2: {
        if (tag == 18) {
         parse_concrete_module:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_concrete_module()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:ElVis.Serialization.RenderModule)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:ElVis.Serialization.RenderModule)
  return false;
#undef DO_
}

void RenderModule::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:ElVis.Serialization.RenderModule)
  // optional bool enabled = 1;
  if (this->enabled() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteBool(1, this->enabled(), output);
  }

  // optional .google.protobuf.Any concrete_module = 2;
  if (this->has_concrete_module()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, *this->concrete_module_, output);
  }

  // @@protoc_insertion_point(serialize_end:ElVis.Serialization.RenderModule)
}

::google::protobuf::uint8* RenderModule::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:ElVis.Serialization.RenderModule)
  // optional bool enabled = 1;
  if (this->enabled() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteBoolToArray(1, this->enabled(), target);
  }

  // optional .google.protobuf.Any concrete_module = 2;
  if (this->has_concrete_module()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        2, *this->concrete_module_, target);
  }

  // @@protoc_insertion_point(serialize_to_array_end:ElVis.Serialization.RenderModule)
  return target;
}

int RenderModule::ByteSize() const {
  int total_size = 0;

  // optional bool enabled = 1;
  if (this->enabled() != 0) {
    total_size += 1 + 1;
  }

  // optional .google.protobuf.Any concrete_module = 2;
  if (this->has_concrete_module()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        *this->concrete_module_);
  }

  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void RenderModule::MergeFrom(const ::google::protobuf::Message& from) {
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  const RenderModule* source = 
      ::google::protobuf::internal::DynamicCastToGenerated<const RenderModule>(
          &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void RenderModule::MergeFrom(const RenderModule& from) {
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  if (from.enabled() != 0) {
    set_enabled(from.enabled());
  }
  if (from.has_concrete_module()) {
    mutable_concrete_module()->::google::protobuf::Any::MergeFrom(from.concrete_module());
  }
}

void RenderModule::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void RenderModule::CopyFrom(const RenderModule& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool RenderModule::IsInitialized() const {

  return true;
}

void RenderModule::Swap(RenderModule* other) {
  if (other == this) return;
  InternalSwap(other);
}
void RenderModule::InternalSwap(RenderModule* other) {
  std::swap(enabled_, other->enabled_);
  std::swap(concrete_module_, other->concrete_module_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata RenderModule::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = RenderModule_descriptor_;
  metadata.reflection = RenderModule_reflection_;
  return metadata;
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// RenderModule

// optional bool enabled = 1;
void RenderModule::clear_enabled() {
  enabled_ = false;
}
 bool RenderModule::enabled() const {
  // @@protoc_insertion_point(field_get:ElVis.Serialization.RenderModule.enabled)
  return enabled_;
}
 void RenderModule::set_enabled(bool value) {
  
  enabled_ = value;
  // @@protoc_insertion_point(field_set:ElVis.Serialization.RenderModule.enabled)
}

// optional .google.protobuf.Any concrete_module = 2;
bool RenderModule::has_concrete_module() const {
  return !_is_default_instance_ && concrete_module_ != NULL;
}
void RenderModule::clear_concrete_module() {
  if (GetArenaNoVirtual() == NULL && concrete_module_ != NULL) delete concrete_module_;
  concrete_module_ = NULL;
}
const ::google::protobuf::Any& RenderModule::concrete_module() const {
  // @@protoc_insertion_point(field_get:ElVis.Serialization.RenderModule.concrete_module)
  return concrete_module_ != NULL ? *concrete_module_ : *default_instance_->concrete_module_;
}
::google::protobuf::Any* RenderModule::mutable_concrete_module() {
  
  if (concrete_module_ == NULL) {
    concrete_module_ = new ::google::protobuf::Any;
  }
  // @@protoc_insertion_point(field_mutable:ElVis.Serialization.RenderModule.concrete_module)
  return concrete_module_;
}
::google::protobuf::Any* RenderModule::release_concrete_module() {
  
  ::google::protobuf::Any* temp = concrete_module_;
  concrete_module_ = NULL;
  return temp;
}
void RenderModule::set_allocated_concrete_module(::google::protobuf::Any* concrete_module) {
  delete concrete_module_;
  concrete_module_ = concrete_module;
  if (concrete_module) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:ElVis.Serialization.RenderModule.concrete_module)
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace Serialization
}  // namespace ElVis

// @@protoc_insertion_point(global_scope)

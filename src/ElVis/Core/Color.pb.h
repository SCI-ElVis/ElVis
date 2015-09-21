// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: Color.proto

#ifndef PROTOBUF_Color_2eproto__INCLUDED
#define PROTOBUF_Color_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3000000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace ElVis {
namespace Serialization {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_Color_2eproto();
void protobuf_AssignDesc_Color_2eproto();
void protobuf_ShutdownFile_Color_2eproto();

class Color;

// ===================================================================

class Color : public ::google::protobuf::Message {
 public:
  Color();
  virtual ~Color();

  Color(const Color& from);

  inline Color& operator=(const Color& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Color& default_instance();

  void Swap(Color* other);

  // implements Message ----------------------------------------------

  inline Color* New() const { return New(NULL); }

  Color* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const Color& from);
  void MergeFrom(const Color& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(Color* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional float red = 1;
  void clear_red();
  static const int kRedFieldNumber = 1;
  float red() const;
  void set_red(float value);

  // optional float green = 2;
  void clear_green();
  static const int kGreenFieldNumber = 2;
  float green() const;
  void set_green(float value);

  // optional float blue = 3;
  void clear_blue();
  static const int kBlueFieldNumber = 3;
  float blue() const;
  void set_blue(float value);

  // optional float alpha = 4;
  void clear_alpha();
  static const int kAlphaFieldNumber = 4;
  float alpha() const;
  void set_alpha(float value);

  // @@protoc_insertion_point(class_scope:ElVis.Serialization.Color)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  bool _is_default_instance_;
  float red_;
  float green_;
  float blue_;
  float alpha_;
  mutable int _cached_size_;
  friend void  protobuf_AddDesc_Color_2eproto();
  friend void protobuf_AssignDesc_Color_2eproto();
  friend void protobuf_ShutdownFile_Color_2eproto();

  void InitAsDefaultInstance();
  static Color* default_instance_;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// Color

// optional float red = 1;
inline void Color::clear_red() {
  red_ = 0;
}
inline float Color::red() const {
  // @@protoc_insertion_point(field_get:ElVis.Serialization.Color.red)
  return red_;
}
inline void Color::set_red(float value) {
  
  red_ = value;
  // @@protoc_insertion_point(field_set:ElVis.Serialization.Color.red)
}

// optional float green = 2;
inline void Color::clear_green() {
  green_ = 0;
}
inline float Color::green() const {
  // @@protoc_insertion_point(field_get:ElVis.Serialization.Color.green)
  return green_;
}
inline void Color::set_green(float value) {
  
  green_ = value;
  // @@protoc_insertion_point(field_set:ElVis.Serialization.Color.green)
}

// optional float blue = 3;
inline void Color::clear_blue() {
  blue_ = 0;
}
inline float Color::blue() const {
  // @@protoc_insertion_point(field_get:ElVis.Serialization.Color.blue)
  return blue_;
}
inline void Color::set_blue(float value) {
  
  blue_ = value;
  // @@protoc_insertion_point(field_set:ElVis.Serialization.Color.blue)
}

// optional float alpha = 4;
inline void Color::clear_alpha() {
  alpha_ = 0;
}
inline float Color::alpha() const {
  // @@protoc_insertion_point(field_get:ElVis.Serialization.Color.alpha)
  return alpha_;
}
inline void Color::set_alpha(float value) {
  
  alpha_ = value;
  // @@protoc_insertion_point(field_set:ElVis.Serialization.Color.alpha)
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace Serialization
}  // namespace ElVis

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_Color_2eproto__INCLUDED
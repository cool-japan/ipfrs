/**
 * @file ipfrs.h
 * @brief IPFRS C API - Foreign Function Interface for InterPlanetary File & Reasoning System
 *
 * This header provides a C-compatible API for IPFRS, enabling integration with
 * C, C++, and other languages that support C FFI.
 *
 * @version 0.1.0
 * @author IPFRS Team
 */

#ifndef IPFRS_H
#define IPFRS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Error codes returned by IPFRS functions
 */
typedef enum {
    /** Operation succeeded */
    IPFRS_SUCCESS = 0,
    /** Null pointer was passed */
    IPFRS_NULL_POINTER = -1,
    /** Invalid UTF-8 string */
    IPFRS_INVALID_UTF8 = -2,
    /** Invalid CID format */
    IPFRS_INVALID_CID = -3,
    /** Block not found */
    IPFRS_NOT_FOUND = -4,
    /** I/O error */
    IPFRS_IO_ERROR = -5,
    /** Out of memory */
    IPFRS_OUT_OF_MEMORY = -6,
    /** Internal error (panic caught) */
    IPFRS_INTERNAL_ERROR = -7,
    /** Invalid argument */
    IPFRS_INVALID_ARGUMENT = -8,
    /** Operation timed out */
    IPFRS_TIMEOUT = -9,
    /** Unknown error */
    IPFRS_UNKNOWN = -99
} IpfrsErrorCode;

/**
 * @brief Opaque handle to IPFRS client
 */
typedef struct IpfrsClient IpfrsClient;

/**
 * @brief Opaque handle to a block
 */
typedef struct IpfrsBlock IpfrsBlock;

/**
 * @brief Initialize a new IPFRS client
 *
 * Creates a new IPFRS client with optional configuration.
 *
 * @param config_path Path to configuration file (can be NULL for defaults)
 * @return Pointer to IpfrsClient on success, NULL on failure
 *
 * @note The returned pointer must be freed with ipfrs_client_free()
 * @note Use ipfrs_get_last_error() to retrieve error message on failure
 *
 * @example
 * @code
 * IpfrsClient* client = ipfrs_client_new(NULL);
 * if (client == NULL) {
 *     fprintf(stderr, "Error: %s\n", ipfrs_get_last_error());
 *     return 1;
 * }
 * // Use client...
 * ipfrs_client_free(client);
 * @endcode
 */
IpfrsClient* ipfrs_client_new(const char* config_path);

/**
 * @brief Free an IPFRS client
 *
 * Releases all resources associated with the client.
 *
 * @param client Pointer to IpfrsClient (must not be NULL)
 *
 * @note The client pointer must not be used after this call
 * @note Passing NULL is a no-op
 */
void ipfrs_client_free(IpfrsClient* client);

/**
 * @brief Add data to IPFRS and return its CID
 *
 * Stores data in IPFRS and returns the Content Identifier (CID).
 *
 * @param client Pointer to IpfrsClient
 * @param data Pointer to data buffer
 * @param data_len Length of data in bytes
 * @param out_cid Output pointer to receive CID string (must be freed with ipfrs_string_free)
 * @return Error code (IPFRS_SUCCESS on success)
 *
 * @note The CID string returned in out_cid must be freed with ipfrs_string_free()
 * @note Use ipfrs_get_last_error() for detailed error information
 *
 * @example
 * @code
 * const char* data = "Hello, IPFRS!";
 * char* cid = NULL;
 * int result = ipfrs_add(client, (const uint8_t*)data, strlen(data), &cid);
 * if (result == IPFRS_SUCCESS) {
 *     printf("CID: %s\n", cid);
 *     ipfrs_string_free(cid);
 * }
 * @endcode
 */
int ipfrs_add(
    IpfrsClient* client,
    const uint8_t* data,
    size_t data_len,
    char** out_cid
);

/**
 * @brief Get data from IPFRS by CID
 *
 * Retrieves data associated with the given CID.
 *
 * @param client Pointer to IpfrsClient
 * @param cid Null-terminated CID string
 * @param out_data Output pointer to receive data buffer (must be freed with ipfrs_data_free)
 * @param out_len Output pointer to receive data length
 * @return Error code (IPFRS_SUCCESS on success)
 *
 * @note The data buffer returned in out_data must be freed with ipfrs_data_free()
 * @note Use ipfrs_get_last_error() for detailed error information
 *
 * @example
 * @code
 * uint8_t* data = NULL;
 * size_t len = 0;
 * int result = ipfrs_get(client, cid, &data, &len);
 * if (result == IPFRS_SUCCESS) {
 *     // Use data...
 *     ipfrs_data_free(data, len);
 * }
 * @endcode
 */
int ipfrs_get(
    IpfrsClient* client,
    const char* cid,
    uint8_t** out_data,
    size_t* out_len
);

/**
 * @brief Check if a block exists by CID
 *
 * Checks whether data with the given CID exists in IPFRS.
 *
 * @param client Pointer to IpfrsClient
 * @param cid Null-terminated CID string
 * @param out_exists Output pointer to receive existence flag (1 = exists, 0 = not found)
 * @return Error code (IPFRS_SUCCESS on success)
 *
 * @note Use ipfrs_get_last_error() for detailed error information
 *
 * @example
 * @code
 * int exists = 0;
 * int result = ipfrs_has(client, cid, &exists);
 * if (result == IPFRS_SUCCESS) {
 *     printf("Block %s: %s\n", cid, exists ? "exists" : "not found");
 * }
 * @endcode
 */
int ipfrs_has(
    IpfrsClient* client,
    const char* cid,
    int* out_exists
);

/**
 * @brief Get the last error message
 *
 * Returns a human-readable description of the last error that occurred.
 *
 * @return Pointer to null-terminated error string, or NULL if no error
 *
 * @note The returned string is valid until the next FFI call on this thread
 * @note DO NOT free the returned pointer
 * @note This function is thread-safe (uses thread-local storage)
 *
 * @example
 * @code
 * if (ipfrs_add(client, data, len, &cid) != IPFRS_SUCCESS) {
 *     fprintf(stderr, "Error: %s\n", ipfrs_get_last_error());
 * }
 * @endcode
 */
const char* ipfrs_get_last_error(void);

/**
 * @brief Free a string returned by IPFRS functions
 *
 * Releases memory allocated for strings returned by IPFRS.
 *
 * @param s Pointer to string (can be NULL)
 *
 * @note The string must have been returned by an IPFRS function
 * @note The string must not be used after this call
 * @note Passing NULL is a no-op
 */
void ipfrs_string_free(char* s);

/**
 * @brief Free data returned by ipfrs_get
 *
 * Releases memory allocated for data buffers.
 *
 * @param data Pointer to data buffer (can be NULL)
 * @param len Length of data buffer (must match the length from ipfrs_get)
 *
 * @note The data must have been returned by ipfrs_get()
 * @note The data must not be used after this call
 * @note Passing NULL for data is a no-op
 */
void ipfrs_data_free(uint8_t* data, size_t len);

/**
 * @brief Get library version string
 *
 * Returns the version of the IPFRS library.
 *
 * @return Pointer to static version string
 *
 * @note DO NOT free the returned pointer
 * @note The returned string is always valid
 *
 * @example
 * @code
 * printf("IPFRS version: %s\n", ipfrs_version());
 * @endcode
 */
const char* ipfrs_version(void);

#ifdef __cplusplus
}
#endif

#endif /* IPFRS_H */

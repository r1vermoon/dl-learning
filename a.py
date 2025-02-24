import heapq
from collections import Counter

# 定义哈夫曼树的节点
class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char  # 字符
        self.freq = freq  # 频率
        self.left = left  # 左子节点
        self.right = right  # 右子节点

    def __lt__(self, other):
        return self.freq < other.freq

# 构建哈夫曼树
def build_huffman_tree(text):
    # 统计字符频率
    frequency = Counter(text)

    # 创建优先队列（最小堆）
    heap = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    # 构建哈夫曼树
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    return heapq.heappop(heap)

# 生成哈夫曼编码
def generate_huffman_codes(node, prefix="", code=None):
    if code is None:
        code = {}
    if node is not None:
        if node.char is not None:
            code[node.char] = prefix
        generate_huffman_codes(node.left, prefix + "0", code)
        generate_huffman_codes(node.right, prefix + "1", code)
    return code

# 哈夫曼编码
def huffman_encode(text):
    if not text:
        return "", None

    root = build_huffman_tree(text)
    codes = generate_huffman_codes(root)
    encoded_text = ''.join([codes[char] for char in text])
    return encoded_text, root

# 哈夫曼解码
def huffman_decode(encoded_text, root):
    if not encoded_text:
        return ""

    decoded_text = []
    current_node = root
    for bit in encoded_text:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        if current_node.char is not None:
            decoded_text.append(current_node.char)
            current_node = root
    return ''.join(decoded_text)

# 测试
if __name__ == "__main__":
    # 输入文本
    text = "A public key infrastructure (PKI) supports the distribution, revocation and verification of public keys used for public key encryption, and enables linking of identities with public key certificates. A PKI enables users and systems to securely exchange data over the internet and verify the legitimacy of certificate-holding entities, such as webservers, other authenticated servers and individuals. The PKI enables users to authenticate digital certificate holders, as well as to mediate the process of certificate revocation, using cryptographic algorithms to secure the process.PKI certificates include a public key used for encryption and cryptographic authentication of data sent to or from the entity that was issued the certificate. Other information included in a PKI certificate includes identifying information about the certificate holder, about the PKI that issued the certificate, and other data including the certificate's creation date and validity period.Without PKI, sensitive information can still be encrypted, ensuring confidentiality, and exchanged between two entities, but there would be no assurance of the identity of the other party. Any form of sensitive data exchanged over the internet is reliant on the PKI for enabling the use of public key cryptography because the PKI enables the authenticated exchange of public keys.Elements of PKIA typical PKI includes the following key elements:A trusted party provides the root of trust for all PKI certificates and provides services that can be used to authenticate the identity of individuals, computers and other entities. Usually known as certificate authorities (CA), these entities provide assurance about the parties identified in a PKI certificate. Each CA maintains its own root CA, for use only by the CA.A registration authority (RA), often called a subordinate CA, issues PKI certificates. The RA is certified by a root CA and authorized to issue certificates for specific uses permitted by the root.A certificate database stores information about issued certificates.In addition to the certificate itself, the database includes validity period and status of each PKI certificate. Certificate revocation is done by updating this database, which must be queried to authenticate any data digitally signed or encrypted with the secret key of the certificate holder.A certificate store, which is usually permanently stored on a computer, can also be maintained in memory for applications that do not require that certificates be stored permanently. The certificate store enables programs running on the system to access stored certificates, certificate revocation lists and certificate trust lists.A CA issues digital certificates to entities and individuals; applicants may be required to verify their identity with increasing degrees of assurance for certificates with increasing levels of validation. The issuing CA digitally signs certificates using its secret key; its public key and digital signature are made available for authentication to all interested parties in a self-signed CA certificate.  "

    # 哈夫曼编码
    encoded_text, tree = huffman_encode(text)
    print("Encoded text:", encoded_text)

    # 哈夫曼解码
    decoded_text = huffman_decode(encoded_text, tree)
    print("Decoded text:", decoded_text)

    # 验证无损压缩
    assert text == decoded_text, "Decoded text does not match the original text!"
    print("Compression and decompression successful!")
---
layout: leetcode-entry
title: "1813. Sentence Similarity III"
permalink: "/leetcode/problem/2024-10-06-1813-sentence-similarity-iii/"
leetcode_ui: true
entry_slug: "2024-10-06-1813-sentence-similarity-iii"
---

[1813. Sentence Similarity III](https://leetcode.com/problems/sentence-similarity-iii/description/) medium
[blog post](https://leetcode.com/problems/sentence-similarity-iii/solutions/5877348/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06102024-1813-sentence-similarity?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/mVnxpNPdCjk)
![1.webp](/assets/leetcode_daily_images/126e7c5e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/758

#### Problem TLDR

Are strings equal after inserting substring? #medium

#### Intuition

The problem becomes easy if we split the words first:

```j

    // a b c d
    // a
    // a d
    // a g d
    // i   j

```
Now, scan prefix words with one pointer `i` and suffix words with another pointer `j`. If `j < i` we good.

The more optimal way, is to not do the splitting: now we have to manually track the space character `' '`, all other logic is the same.

#### Approach

* split words for shorter code
* to track the word breaks, consider checking a single out of boundary position as a space char `' '`.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$ or O(n) for word split

#### Code

```kotlin

    fun areSentencesSimilar(sentence1: String, sentence2: String): Boolean {
        val words1 = sentence1.split(" "); val words2 = sentence2.split(" ")
        var i = 0; var j1 = words1.lastIndex; var j2 = words2.lastIndex
        while (i < words1.size && i < words2.size && words1[i] == words2[i]) i++
        while (j1 >= i && j2 >= i && words1[j1] == words2[j2]) { j1--; j2-- }
        return j1 < i || j2 < i
    }

```
```rust

    pub fn are_sentences_similar(sentence1: String, sentence2: String) -> bool {
        let (bytes1, bytes2) = (sentence1.as_bytes(), sentence2.as_bytes());
        let (n1, n2) = (bytes1.len(), bytes2.len());
        let (mut i, mut j, mut k, mut k1, mut k2) = (0, 0, 0, n1 as i32 - 1, n2 as i32 - 1);
        while k <= n1 && k <= n2 {
            let a = if k == n1 { b' ' } else { bytes1[k] };
            let b = if k == n2 { b' ' } else { bytes2[k] };
            if a != b { break }; if a == b' ' { i += 1 }; k += 1
        }
        while k1 >= -1 && k2 >= -1 {
            let a = if k1 < 0 { b' ' } else { bytes1[k1 as usize] };
            let b = if k2 < 0 { b' ' } else { bytes2[k2 as usize] };
            if a != b { break }; if a == b' ' { j += 1 }; k1 -= 1; k2 -= 1
        }
        bytes1.iter().filter(|&&b| b == b' ').count() as i32 - j < i ||
        bytes2.iter().filter(|&&b| b == b' ').count() as i32 - j < i
    }

```
```c++

    bool areSentencesSimilar(string sentence1, string sentence2) {
        int i = 0, j = 0, k = 0, k1 = sentence1.length() - 1, k2 = sentence2.length() - 1;
        while (k <= sentence1.length() && k <= sentence2.length()) {
            char a = k == sentence1.length() ? ' ' : sentence1[k];
            char b = k == sentence2.length() ? ' ' : sentence2[k];
            if (a != b) break; if (a == ' ') i++; k++;
        }
        while (k1 >= -1 && k2 >= -1) {
            char a = k1 < 0 ? ' ' : sentence1[k1];
            char b = k2 < 0 ? ' ' : sentence2[k2];
            if (a != b) break; if (a == ' ') j++; k1--; k2--;
        }
        return ranges::count(sentence1, ' ') - j < i || ranges::count(sentence2, ' ') - j < i;
    }

```


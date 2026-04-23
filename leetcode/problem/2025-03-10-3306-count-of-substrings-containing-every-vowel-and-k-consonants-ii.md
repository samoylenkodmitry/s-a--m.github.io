---
layout: leetcode-entry
title: "3306. Count of Substrings Containing Every Vowel and K Consonants II"
permalink: "/leetcode/problem/2025-03-10-3306-count-of-substrings-containing-every-vowel-and-k-consonants-ii/"
leetcode_ui: true
entry_slug: "2025-03-10-3306-count-of-substrings-containing-every-vowel-and-k-consonants-ii"
---

[3306. Count of Substrings Containing Every Vowel and K Consonants II](https://leetcode.com/problems/count-of-substrings-containing-every-vowel-and-k-consonants-ii/description/) medium
[blog post](https://leetcode.com/problems/count-of-substrings-containing-every-vowel-and-k-consonants-ii/solutions/6519609/kotlin-rust-by-samoylenkodmitry-arzf/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10032025-3306-count-of-substrings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/aecs4eghif4)
![1.webp](/assets/leetcode_daily_images/ca3db59d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/921

#### Problem TLDR

Substring with all vowels and k others #medium #two_pointers

#### Intuition

The naive two pointers would not work for the case of repeating suffixes and prefixes:

```j

"iiiiiqeaouqi" k = 2

```
So, we should somehow track it.
Let's introduce the third pointer b (Rust solution): `border at which we have minimum vowels and k others`.

Another approach is the trick: `k = atLeast(k) - atLeast(k + 1)`. (At `most` wouldn't work though)

#### Approach

* trick from `u/votrubac/`: use indexOf to cleverly store both vowels and not vowels

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countOfSubstrings(w: String, k: Int): Long {
        fun atLeast(k: Int): Long {
            var r = 0L; var j = 0; val cnt = IntArray(6); var u = 0;
            for (i in w.indices) {
                val p = "aeiou".indexOf(w[i]) + 1; if (cnt[p]++ < 1 && p > 0) u++
                while (u == 5 && cnt[0] >= k) {
                    r += w.length - i
                    val p = "aeiou".indexOf(w[j++]) + 1; if (--cnt[p] < 1 && p > 0) u--
                }
            }
            return r
        }
        return atLeast(k) - atLeast(k + 1)
    }

```
```rust

    pub fn count_of_substrings(word: String, k: i32) -> i64 {
        let w = word.as_bytes(); let wv = |b| (1065233 >> (b - b'a') & 1) > 0;
        let (mut cw, mut cc, mut fw, mut bw, mut b, mut fb, mut j) = (0, 0, vec![0; 26], 0, 0, vec![0; 26], 0);
        (0..w.len()).map(|i|{
            if wv(w[i]) { let i = (w[i] - b'a') as usize;
                if fb[i] < 1 { bw += 1 }; fb[i] += 1; if fw[i] < 1 { cw += 1 }; fw[i] += 1;
            } else { cc += 1 }
            while cc > k {
                if wv(w[j]) { let wj = (w[j] - b'a') as usize; if fw[wj] == 1 { cw -= 1 }; fw[wj] -= 1;
                } else { cc -= 1 }
                j += 1
            }
            while b < j || b < w.len() && cc == k && fb[(w[b] - b'a') as usize] > 1 {
                let wb = (w[b] - b'a') as usize; if fb[wb] == 1 { bw -= 1 }; fb[wb] -= 1; b += 1
            }
            if cw == 5 && cc == k { 1 + b as i64 - j as i64 } else { 0 }
        }).sum()
    }

```
```c++

    long countOfSubstrings(const string &w, int k) {
        string vw = "aeiou"; auto atLeast = [&](int k) {
            long r = 0; int j = 0, u = 0, cnt[6] = {};
            for (int i = 0; i < w.size(); i++) {
                int p = vw.find(w[i]) + 1;
                u += ++cnt[p] == 1 && p;
                while (u == 5 && cnt[0] >= k) {
                    r += w.size() - i;
                    int q = vw.find(w[j++]) + 1;
                    u -= --cnt[q] == 0 && q;
                }
            }
            return r;
        };
        return atLeast(k) - atLeast(k + 1);
    }

```


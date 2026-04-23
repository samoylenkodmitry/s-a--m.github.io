---
layout: leetcode-entry
title: "2182. Construct String With Repeat Limit"
permalink: "/leetcode/problem/2024-12-17-2182-construct-string-with-repeat-limit/"
leetcode_ui: true
entry_slug: "2024-12-17-2182-construct-string-with-repeat-limit"
---

[2182. Construct String With Repeat Limit](https://leetcode.com/problems/construct-string-with-repeat-limit/description/) medium
[blog post](https://leetcode.com/problems/construct-string-with-repeat-limit/solutions/6155550/kotlin-rust-by-samoylenkodmitry-8frw/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17122024-2182-construct-string-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tGaNBtqpNec)
[deep-dive](https://notebooklm.google.com/notebook/a42da74e-c8c3-4d22-942b-93765837ccee/audio)
![1.webp](/assets/leetcode_daily_images/5d21bf37.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/835

#### Problem TLDR

Max lexical ordered with `repeat_limit` string #medium #bucket_sort

#### Intuition

Always peek the largest value. If limit is reached peek one of the next.

#### Approach

* we can use a heap, but have to manage all the next chars at once
* we can use a frequency counter and two pointers: current and next

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun repeatLimitedString(s: String, repeatLimit: Int) = buildString {
        val f = IntArray(26); for (c in s) f[c - 'a']++
        var cnt = 0; var i = 25; var j = i
        while (i >= 0) {
            if (f[i] == 0) { i--; continue }
            if (length == 0 || get(length - 1) == 'a' + i) cnt++ else cnt = 1
            if (cnt > repeatLimit) {
                j = min(j, i - 1); while (j >= 0 && f[j] == 0) j--
                if (j >= 0) { append('a' + j); f[j]-- } else break
            } else { append('a' + i); f[i]-- }
        }
    }

```
```rust

    pub fn repeat_limited_string(s: String, repeat_limit: i32) -> String {
        let mut f = [0; 26]; for b in s.bytes() { f[(b - b'a') as usize] += 1 }
        let (mut cnt, mut i, mut j, mut r) = (0, 25, 25, vec![]);
        loop {
            if f[i] == 0 { if i == 0 { break } else { i -= 1; continue }}
            if r.last().map_or(true, |&l| l == b'a' + i as u8) { cnt += 1 } else { cnt = 1 }
            if cnt > repeat_limit {
                if i == 0 { break } else { j = j.min(i - 1) }
                loop { if j == 0 || f[j] > 0 { break } else { j -= 1 }}
                if f[j] > 0 { r.push(b'a' + j as u8); f[j] -= 1 } else { break }
            } else { r.push(b'a' + i as u8); f[i] -= 1}
        }; String::from_utf8(r).unwrap()
    }

```
```c++

    string repeatLimitedString(string s, int repeatLimit) {
        int f[26]; for (auto c: s) ++f[c - 'a'];
        int cnt = 0, i = 25, j = 25, l = 26; string r;
        while (i >= 0) {
            if (!f[i]) { --i; continue; }
            l == i ? ++cnt : cnt = 1;
            if (cnt > repeatLimit) {
                j = min(j, i - 1);
                while (j > 0 && !f[j]) --j;
                if (j >= 0 && f[j]--) { r += 'a' + j; l = j; } else break;
            } else { r += 'a' + i; --f[i]; l = i; }
        } return r;
    }

```


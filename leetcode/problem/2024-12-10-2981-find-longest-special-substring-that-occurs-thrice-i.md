---
layout: leetcode-entry
title: "2981. Find Longest Special Substring That Occurs Thrice I"
permalink: "/leetcode/problem/2024-12-10-2981-find-longest-special-substring-that-occurs-thrice-i/"
leetcode_ui: true
entry_slug: "2024-12-10-2981-find-longest-special-substring-that-occurs-thrice-i"
---

[2981. Find Longest Special Substring That Occurs Thrice I](https://leetcode.com/problems/find-longest-special-substring-that-occurs-thrice-i/description/) medium
[blog post](https://leetcode.com/problems/find-longest-special-substring-that-occurs-thrice-i/solutions/6131875/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10122024-2981-find-longest-special?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/duSxeDfaF2w)
[deep-dive](https://notebooklm.google.com/notebook/5405ba0d-ea5b-4fb5-b4ff-c3809d90503b/audio)
![1.jpg](/assets/leetcode_daily_images/76520683.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/828

#### Problem TLDR

Max same-char 3-windows length #medium #sliding_window

#### Intuition

Problem size is small, brute force works: try every length, do a sliding window.

Slightly better is to do a binary search of window length.

The clever solution is to precompute window frequencies and then check every length:

```j

`aaaa` -> 'a' -> 1, 'aa' -> 1, 'aaa' -> 1, 'aaaa' -> 1

len: 4 -> 1, 3 -> f(4) + 1 = 2, 2 -> f(3) + 1 = 3 (take)

```

#### Approach

* try every approach

#### Complexity

- Time complexity:
$$O(n^2)$$ -> O(nlog(n)) -> O(n)

- Space complexity:
$$O(n)$$ -> O(1)

#### Code

```kotlin

    fun maximumLength(s: String) =
        (s.length - 2 downTo 1).firstOrNull { len ->
            val f = IntArray(128)
            s.windowed(len).any { w -> w.all { it == w[0] } && ++f[w[0].code] > 2 }
        } ?: -1

```

```rust

    pub fn maximum_length(s: String) -> i32 {
        let (mut lo, mut hi, b, mut r) = (1, s.len() - 2, s.as_bytes(), -1);
        while lo <= hi {
            let m = lo + (hi - lo) / 2; let mut f = vec![0; 26];
            if b[..].windows(m).any(|w|
                w.iter().all(|&x| x == w[0]) && {
                f[(w[0] - b'a') as usize] += 1; f[(w[0] - b'a') as usize] > 2
            }) { r = r.max(m as i32); lo = m + 1 } else { hi = m - 1 }
        }; r
    }

```

```c++

    int maximumLength(string s) {
        vector<vector<int>> f(26, vector<int>(s.size() + 1, 0));
        char p = '.'; int cnt = 0, res = -1;
        for (auto c: s) f[c - 'a'][c == p ? ++cnt : (cnt = 1)]++, p = c;
        for (int c = 0; c < 26; ++c)
            for (int l = s.size(), p = 0; l; --l)
                if ((p += f[c][l]) > 2) { res = max(res, l); break; }
        return res;
    }

```


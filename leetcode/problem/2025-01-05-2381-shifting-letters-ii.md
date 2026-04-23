---
layout: leetcode-entry
title: "2381. Shifting Letters II"
permalink: "/leetcode/problem/2025-01-05-2381-shifting-letters-ii/"
leetcode_ui: true
entry_slug: "2025-01-05-2381-shifting-letters-ii"
---

[2381. Shifting Letters II](https://leetcode.com/problems/shifting-letters-ii/description/) medium
[blog post](https://leetcode.com/problems/shifting-letters-ii/solutions/6233547/kotlin-rust-by-samoylenkodmitry-f183/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05012025-2381-shifting-letters-ii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/i35fgm4-epw)
[deep-dive](https://notebooklm.google.com/notebook/ec031256-a0ee-454c-8514-b1051e6ed029/audio)
![1.webp](/assets/leetcode_daily_images/083f325e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/855

#### Problem TLDR

Apply `from..to, direction` shifts to string chars #medium #line_sweep

#### Intuition

We can sort the shifts intervals, then walk them, calculating the running value of shift.

One optimization is to store the starts and ends of each shift in a cumulative shifts array, then scan it's running value in a linear way.

#### Approach

* in Rust we can modify the stirng in-place with `unsafe { s.as_bytes_mut() }`
* the difference betwen `auto s: shifts` and `auto &s: shifts` in c++ is 4ms vs 40ms

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun shiftingLetters(s: String, shifts: Array<IntArray>) = buildString {
        val shift = IntArray(s.length + 1); var sh = 0
        for ((s, e, d) in shifts) {
            shift[s] += d * 2 - 1
            shift[e + 1] -= d * 2 - 1
        }
        for ((i, c) in s.withIndex()) {
            sh += shift[i]
            append('a' + (c - 'a' + sh % 26 + 26) % 26)
        }
    }

```
```rust

    pub fn shifting_letters(s: String, shifts: Vec<Vec<i32>>) -> String {
        let (mut shift, mut sh, mut r) = (vec![0; s.len() + 1], 0, vec![0; s.len()]);
        for sh in shifts {
            let (s, e, d) = (sh[0] as usize, sh[1] as usize, sh[2] * 2 - 1);
            shift[s] += d; shift[e + 1] -= d
        }
        for (i, c) in s.bytes().enumerate() {
            sh += shift[i];
            r[i] = b'a' + (c - b'a' + (sh % 26 + 26) as u8) % 26
        }; String::from_utf8(r).unwrap()
    }

```
```c++

    string shiftingLetters(string s, vector<vector<int>>& shifts) {
        int sh[50001] = {0}, d = 0;
        for (auto &s: shifts)
            sh[s[0]] += s[2] * 2 - 1, sh[s[1] + 1] -= s[2] * 2 - 1;
        for (int i = 0; i < s.size(); ++i)
            s[i] = 'a' + (s[i] - 'a' + (d += sh[i]) % 26 + 26) % 26;
        return s;
    }

```


---
layout: leetcode-entry
title: "2138. Divide a String Into Groups of Size k"
permalink: "/leetcode/problem/2025-06-22-2138-divide-a-string-into-groups-of-size-k/"
leetcode_ui: true
entry_slug: "2025-06-22-2138-divide-a-string-into-groups-of-size-k"
---

[2138. Divide a String Into Groups of Size k](https://leetcode.com/problems/divide-a-string-into-groups-of-size-k/description/) easy
[blog post](https://leetcode.com/problems/divide-a-string-into-groups-of-size-k/solutions/6871846/kotlin-rust-by-samoylenkodmitry-ajm8/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22062025-2138-divide-a-string-into?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/TUiVCSC3H_I)
![1.webp](/assets/leetcode_daily_images/19932874.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1027

#### Problem TLDR

k-chunked string, filled tail #easy

#### Intuition

Pad then chunk, or chunk then pad. Prefill everything, or write a precise filling code. Great task to learn the language built-ins.

#### Approach

* if you know Kotlin `padEnd` & `chunked` you don't have to think
* Rust doesn't allow `fmt` with dynamic fill character
* `1 + (size - 1) / k` or `(k + size - 1) / k` will pad to `% k`
* Kotln / Java `String` has `CharArray` constructor arguments
* Rust has `resize` to pad-fill a `Vec`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 14ms
    fun divideString(s: String, k: Int, fill: Char) =
        s.chunked(k).map { it.padEnd(k, fill) }

```
```kotlin

// 9ms
    fun divideString(s: String, k: Int, fill: Char) =
        Array(1 + (s.length - 1) / k) {  s.drop(it * k).take(k).padEnd(k, fill) }

```
```kotlin

// 8ms
    fun divideString(s: String, k: Int, fill: Char) =
        s.padEnd((1 + (s.length - 1) / k) * k, fill).chunked(k)

```
```kotlin

// 1ms
    fun divideString(s: String, k: Int, fill: Char): Array<String> {
        val s = s.toCharArray()
        return Array(1 + (s.size - 1) / k) { i ->
            val sz = min(s.size - i * k, k)
            if (sz == k) String(s, i * k, k)
            else {
                val tmp = CharArray(k) { fill };
                System.arraycopy(s, i * k, tmp, 0, sz)
                String(tmp)
            }
        }
    }

```

```rust

// 0ms
    pub fn divide_string(s: String, k: i32, fill: char) -> Vec<String> {
        let mut s = s.chars().collect::<Vec<_>>(); let k = k as usize;
        s.resize((1 + (s.len() - 1) / k) * k, fill);
        s.chunks(k).map(|c| c.iter().collect()).collect()
    }

```

```c++

// 0ms
    vector<string> divideString(string s, int k, char fill) {
        vector<string> r(1 + (size(s) - 1) / k, string(k, fill));
        for (int i = 0; i < size(s); ++i) r[i / k][i % k] = s[i];
        return r;
    }

```


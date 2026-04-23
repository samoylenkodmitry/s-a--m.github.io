---
layout: leetcode-entry
title: "869. Reordered Power of 2"
permalink: "/leetcode/problem/2025-08-10-869-reordered-power-of-2/"
leetcode_ui: true
entry_slug: "2025-08-10-869-reordered-power-of-2"
---

[869. Reordered Power of 2](https://leetcode.com/problems/reordered-power-of-2/description/) medium
[blog post](https://leetcode.com/problems/reordered-power-of-2/solutions/7063607/kotlin-rust-by-samoylenkodmitry-u5zx/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10082025-869-reordered-power-of-2?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/uPz9eLlQv2s)
![1.webp](/assets/leetcode_daily_images/095a98ef.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1076

#### Problem TLDR

Rearrange digits to make power of two #medium #dfs #bits

#### Intuition

The naive brute force of `O(log(n)!)` is accepted, we only have `9` digits, n is 10^9.

The trick: think from the finish line - we only have `30` powers of two, let's just check them.

#### Approach

* compare by frequency
* optimization: for `9` digits, each max frequency is `9`, we can put freqency array into a single `10` digits number (c++ solution)

#### Complexity

- Time complexity:
$$O(log^2(n))$$,

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin [-Kotlin (233ms]

// 233ms
    fun reorderedPowerOf2(n: Int): Boolean {
        val ds = "$n"
        fun dfs(curr: Int, m: Int): Boolean {
            if (m == 0) return curr.countOneBits() == 1
            for (i in 0..30) if (m and (1 shl i) != 0) {
                if (ds[i] == '0' && curr == 0) continue
                if (dfs(curr * 10 + (ds[i] - '0'), m xor (1 shl i))) return true
            }
            return false
        }
        return dfs(0, (1 shl ds.length) - 1)
    }

```
```kotlin [-18ms)]

// 18ms
    fun reorderedPowerOf2(n: Int) =
        (0..30).any { "$n".groupBy { it } == "${1 shl it}".groupBy { it }}

```
```rust [-Rust (0ms]

// 0ms
    pub fn reordered_power_of2(mut n: i32) -> bool {
        let mut nf = [0; 10]; while n > 0 { nf[(n % 10) as usize] += 1; n /= 10 }
        (0..30).any(|i| {
            let mut x = 1 << i; let mut f = [0; 10];
            while x > 0 { f[(x % 10) as usize] += 1; x /= 10 }
            f == nf
        })
    }

```
```c++ [-c++ (0ms]

// 0ms
    bool reorderedPowerOf2(int n) {
        long c = 0; while (n) c += pow(10, n % 10), n /= 10;
        for (int i = 0, x=0, y=1; i < 30; ++i, x = 0, y = 1<<i) {
            while (y) x += pow(10, y % 10), y /= 10;
            if (x == c) return 1;
        }
        return 0;
    }

```
```python [-python 0ms]

// 0ms
    def reorderedPowerOf2(self, n: int) -> bool:
        return any(sorted(str(n)) == sorted(str(1 << i)) for i in range(31))

```

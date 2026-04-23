---
layout: leetcode-entry
title: "118. Pascal's Triangle"
permalink: "/leetcode/problem/2025-08-01-118-pascal-s-triangle/"
leetcode_ui: true
entry_slug: "2025-08-01-118-pascal-s-triangle"
---

[118. Pascal's Triangle](https://leetcode.com/problems/pascals-triangle/description/) easy
[blog post](https://leetcode.com/problems/pascals-triangle/solutions/7031375/kotlin-rust-by-samoylenkodmitry-hbpk/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/1082025-118-pascals-triangle?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/TPNlXnW3aM8)
![1.webp](/assets/leetcode_daily_images/fcea5ea0.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1067

#### Problem TLDR

Pascal's Triangle #easy

#### Intuition

Classic problem, reuse the previous row.

#### Approach

* many ways to write this: fold, scan, recursion, zip

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(2^n)$$

#### Code

```kotlin

// 15ms
    fun generate(n: Int) = (2..n).runningFold(listOf(1))
    { r, t -> listOf(1) + r.windowed(2) { it.sum() } + 1 }

```

```rust

// 0ms
    pub fn generate(n: i32) -> Vec<Vec<i32>> {
        (0..n).scan(vec![1], |c, _| { let r = c.clone();
            *c = vec![vec![1], c.windows(2).map(|w| w[0] + w[1]).collect(), vec![1]].concat();
            Some(r)
        }).collect()
    }

```
```c++

// 0ms
    vector<vector<int>> generate(int n) {
        if (n == 1) return {{1}}; auto p = generate(n - 1); vector<int>r{1};
        for (int i = 1; i < size(p[n - 2]); ++i)
            r.push_back(p[n - 2][i - 1] + p[n - 2][i]);
        r.push_back(1); p.push_back(r); return p;
    }

```
```python3

// 0ms
    def generate(self, n: int) -> List[List[int]]:
        r=[]
        for _ in[0]*n:r+=[[1]]if not r else[[1]+[a+b for a,b in zip(r[-1],r[-1][1:])]+[1]]
        return r

```


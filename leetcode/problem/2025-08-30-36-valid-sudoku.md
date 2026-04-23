---
layout: leetcode-entry
title: "36. Valid Sudoku"
permalink: "/leetcode/problem/2025-08-30-36-valid-sudoku/"
leetcode_ui: true
entry_slug: "2025-08-30-36-valid-sudoku"
---

[36. Valid Sudoku](https://leetcode.com/problems/valid-sudoku/description/) medium
[blog post](https://leetcode.com/problems/valid-sudoku/solutions/7137955/kotlin-rust-by-samoylenkodmitry-uktg/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30082025-36-valid-sudoku?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XSLc3mXzOuQ)

![1.webp](/assets/leetcode_daily_images/98833e5d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1097

#### Problem TLDR

Validate sudoku has no duplicates #medium

#### Intuition

Brute-force.

#### Approach

* small grid is `big * 3 + small`
* single hashset: use keys as `row + digit`, `column + digit`, `box + digit`

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 0ms
    fun List<Char>.ok() = filter { it != '.' }.let { it.toSet().size == it.size }
    fun isValidSudoku(b: Array<CharArray>) =
        (0..8).all { y -> b[y].map { it }.ok() } &&
        (0..8).all { x -> (0..8).map { b[it][x] }.ok() } &&
        (0..8).all { c -> (0..8).map { b[c/3 * 3 + it/3][c%3 * 3 + it%3]}.ok() }

```
```rust

// 0ms
    pub fn is_valid_sudoku(b: Vec<Vec<char>>) -> bool {
        let (mut cols, mut rows, mut subs) = ([0;9],[0;9],[0;9]);
        for y in 0..9 { for x in 0..9 { if b[y][x] != '.' {
            let d = 1 << (b[y][x] as u8 - b'1');
            if (cols[x] & d) + (rows[y] & d) + (subs[y/3*3+x/3] & d) > 0 { return false }
            cols[x] |= d; rows[y] |= d; subs[y/3*3+x/3] |= d;
        }}} true
    }

```
```c++

// 0ms
    bool isValidSudoku(vector<vector<char>>& b) {
        int f[244]={};
        for (int y = 0; y < 9; ++y) for (int x = 0; x < 9; ++x) if (b[y][x] != '.') {
            int d = b[y][x] - '1';
            if (f[y*9+d]+f[81+d*9+x]+f[81+81+(y/3*3+x/3)*9+d]) return 0;
            f[y*9+d]=1;f[81+d*9+x]=1;f[81+81+(y/3*3+x/3)*9+d]=1;
        } return 1;
    }

```
```python

// 5ms
    def isValidSudoku(_, b):
        a = sum(([(d,y),(x,d),(y//3,x//3,d)]
                for y in range(9) for x in range(9)
                for d in [b[y][x]] if d != '.'),[])
        return len(a) == len(set(a))

```


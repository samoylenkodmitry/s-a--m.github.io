---
layout: leetcode-entry
title: "2337. Move Pieces to Obtain a String"
permalink: "/leetcode/problem/2024-12-05-2337-move-pieces-to-obtain-a-string/"
leetcode_ui: true
entry_slug: "2024-12-05-2337-move-pieces-to-obtain-a-string"
---

[2337. Move Pieces to Obtain a String](https://leetcode.com/problems/move-pieces-to-obtain-a-string/description/) medium
[blog post](https://leetcode.com/problems/move-pieces-to-obtain-a-string/solutions/6115673/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05122024-2337-move-pieces-to-obtain?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UijiTqgpfOo)
[deep-dive](https://notebooklm.google.com/notebook/cb694c34-19bb-4a2b-ab72-0054c83fc9e4/audio)
![1.webp](/assets/leetcode_daily_images/16442bf8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/823

#### Problem TLDR

Move `L` left and `R` right to match strings #medium

#### Intuition

Let's move both pointers together and calculate the balance of `L`, `R` and `_`:

```j

    // R_R___L  __RRL__  R       b       L
    // j      //   jk
    //  j   .    i   .           +1-1=1
    //   j  .     i  .   +1-1=1
    //    j .      i .   -1=0    +1=2
    //     j.       i.           +1=3     -1 (check R==0)
    //      j        i           +1-1=3
    //       j        i          -1=2     +1=0

```

Some observations:
* the final balance should be `0`
* we should eliminate the impossible scenarios (that's where the hardness of this task begins)
* to simplify the corner cases let's split this into pass forward and pass backwards (then we have ugly long solution but its werks)

Now, the more clever way of solving ignore the spaces `_` and only check the balance of `l` and `r` be not negative. The corner case would be `LR` -> `RL` and for this check we don't have `l > 0` and `r > 0` together.

Another, much simpler way of thinking: move separate pointers instead of a single, and skip the spaces `_`, then compare:
* `s[i] == t[j]` letters should match
* some indexes rules: from `start` `R` goes forward `i <= j`, `L` goes backward `i >= j`

#### Approach

* slow down and think one step at a time
* the good idea of separate pointers eliminates all corner cases (so think broader in a space of ideas before thinking in a space of implementations)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun canChange(start: String, target: String): Boolean {
        var l = 0; var r = 0
        for ((i, s) in start.withIndex()) {
            val t = target[i]
            if (s == 'R') r++
            if (t == 'L') l++
            if (l * r > 0) return false
                       kk
            if (s == 'L' && --l < 0) return false
           k
        return l == 0 && r == 0
    }

```
```rust

    pub fn can_change(start: String, target: String) -> bool {
        let (mut i, mut j, s, t, n) =
            (0, 0, start.as_bytes(), target.as_bytes(), start.len());
        while i < n || j < n {
            while i < n && s[i] == b'_' { i += 1 }
            while j < n && t[j] == b'_' { j += 1 }
            if i == n || j == n || s[i] != t[j] ||
            s[i] == b'L' && i < j || s[i] == b'R' && i > j { break }
            i += 1; j += 1
        }; i == n && j == n
    }

```
```c++

    bool canChange(string s, string t) {
        int l = 0, r = 0;
        for (int i = 0; i < s.size(); ++i) {
            if (s[i] == 'R') r++;
            if (t[i] == 'L') l++;
            if (l * r > 0) return 0;
            if (t[i] == 'R' && --r < 0) return 0;
            if (s[i] == 'L' && --l < 0) return 0;
        }
        return l == 0 && r == 0;
    }

```

